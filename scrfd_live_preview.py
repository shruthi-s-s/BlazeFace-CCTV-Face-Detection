

from pathlib import Path
from collections import defaultdict
from fractions import Fraction

import csv
import json
import queue
import threading
import time
import tkinter as tk

import av
import numpy as np
import onnxruntime as ort

from PIL import Image, ImageDraw, ImageTk
from scipy.optimize import linear_sum_assignment


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE = Path(__file__).resolve().parent

VIDEO_PATH = BASE / "videos" / "input.mp4"

MODEL_CANDIDATES = [
    BASE / "models" / "2.5g_bnkps.onnx",
    BASE / "models" / "scrfd_2.5g_bnkps.onnx",
]

MODEL_PATH = next(
    (p for p in MODEL_CANDIDATES if p.is_file()),
    MODEL_CANDIDATES[0]
)

OUTPUT_DIR = BASE / "videos" / "live_results"

MODEL_SIZE = 640

HIGH_CONF = 0.55
LOW_CONF = 0.25

NMS_THRESHOLD = 0.40

# Detection every 2 frames.
# Tracking still processes EVERY frame.
# Change to 1 for detection on every frame.
DETECTION_INTERVAL = 2

MIN_TRACK_HITS = 3
MAX_LOST_FRAMES = 36
MAX_VISIBLE_MISSED = 3

MIN_FACE_SIZE = 48
MIN_BLUR_VARIANCE = 55.0

MIN_BRIGHTNESS = 45
MAX_BRIGHTNESS = 215

BEST_FRAMES_PER_TRACK = 5
MIN_BEST_FRAME_GAP = 5

PREVIEW_MAX_WIDTH = 1100

# Optional region of interest:
# None = entire video
# Example: (100, 100, 1100, 650)
ROI = None

ENABLE_ADAPTIVE_TILING = False
TILE_SIZE = 640
TILE_OVERLAP = 0.25
MAX_TILES = 6

ENABLE_CLAHE = False

# Process sample video once.
LOOP_VIDEO = False

# Enable if you need annotated MP4.
SAVE_ANNOTATED_VIDEO = False

# IMPORTANT:
# This version NEVER intentionally skips
# a decoded video frame.
SKIP_FRAMES = False


# ==========================================
# 2. BOX UTILITIES
# ==========================================

def calculate_iou(a, b):

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])

    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )

    area_a = (
        max(0, a[2] - a[0])
        * max(0, a[3] - a[1])
    )

    area_b = (
        max(0, b[2] - b[0])
        * max(0, b[3] - b[1])
    )

    return intersection / max(
        area_a + area_b - intersection,
        1e-6
    )


def inside_roi(box):

    if ROI is None:
        return True

    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2

    return (
        ROI[0] <= cx <= ROI[2]
        and ROI[1] <= cy <= ROI[3]
    )


def non_max_suppression(detections):

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda item: item[1],
        reverse=True
    )

    selected = []

    while detections:

        best = detections.pop(0)
        selected.append(best)

        detections = [
            item
            for item in detections
            if calculate_iou(
                best[0],
                item[0]
            ) < NMS_THRESHOLD
        ]

    return selected


# ==========================================
# 3. SCRFD PREPROCESSING
# ==========================================

def preprocess_frame(frame_array):

    height, width = frame_array.shape[:2]

    scale = min(
        MODEL_SIZE / width,
        MODEL_SIZE / height
    )

    new_width = max(
        1,
        round(width * scale)
    )

    new_height = max(
        1,
        round(height * scale)
    )

    temporary_frame = av.VideoFrame.from_ndarray(
        np.ascontiguousarray(frame_array),
        format="bgr24"
    )

    resized = temporary_frame.reformat(
        width=new_width,
        height=new_height,
        format="bgr24"
    ).to_ndarray()

    canvas = np.zeros(
        (MODEL_SIZE, MODEL_SIZE, 3),
        dtype=np.uint8
    )

    canvas[:new_height, :new_width] = resized

    tensor = (
        canvas.astype(np.float32) - 127.5
    ) / 128.0

    tensor = np.ascontiguousarray(
        tensor.transpose(2, 0, 1)[None]
    )

    return tensor, scale


# ==========================================
# 4. SCRFD DETECTOR
# ==========================================

class SCRFDDetector:

    def __init__(self):

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        options = ort.SessionOptions()

        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = (
            self.session.get_inputs()[0].name
        )

        self.output_names = [
            output.name
            for output in self.session.get_outputs()
        ]

        self.inference_count = 0
        self.inference_time = 0.0

    def detect_array(
        self,
        frame_array,
        offset=(0, 0)
    ):

        tensor, scale = preprocess_frame(
            frame_array
        )

        start = time.perf_counter()

        raw_outputs = self.session.run(
            None,
            {self.input_name: tensor}
        )

        self.inference_time += (
            time.perf_counter() - start
        )

        self.inference_count += 1

        outputs = dict(
            zip(
                self.output_names,
                raw_outputs
            )
        )

        detections = []

        for stride in (8, 16, 32):

            scores = outputs[
                f"score_{stride}"
            ][0].reshape(-1)

            bbox = outputs[
                f"bbox_{stride}"
            ][0].reshape(-1, 4)

            grid_size = MODEL_SIZE // stride

            yy, xx = np.mgrid[
                :grid_size,
                :grid_size
            ]

            centers = np.stack(
                (xx, yy),
                axis=-1
            ).reshape(-1, 2)

            anchors_per_cell = (
                len(scores) // len(centers)
            )

            centers = np.repeat(
                centers,
                anchors_per_cell,
                axis=0
            )

            centers = (
                centers.astype(np.float32)
                * stride
            )

            selected = np.where(
                scores >= LOW_CONF
            )[0]

            if len(selected) == 0:
                continue

            distances = (
                bbox[selected] * stride
            )

            center = centers[selected]

            boxes = np.column_stack([
                center[:, 0] - distances[:, 0],
                center[:, 1] - distances[:, 1],
                center[:, 0] + distances[:, 2],
                center[:, 1] + distances[:, 3]
            ])

            boxes /= scale

            boxes[:, [0, 2]] = np.clip(
                boxes[:, [0, 2]],
                0,
                frame_array.shape[1]
            )

            boxes[:, [1, 3]] = np.clip(
                boxes[:, [1, 3]],
                0,
                frame_array.shape[0]
            )

            boxes[:, [0, 2]] += offset[0]
            boxes[:, [1, 3]] += offset[1]

            for box, confidence in zip(
                boxes,
                scores[selected]
            ):

                if (
                    box[2] > box[0]
                    and box[3] > box[1]
                    and inside_roi(box)
                ):

                    detections.append((
                        box.astype(np.float64),
                        float(confidence)
                    ))

        return detections

    def detect(self, frame):

        frame_array = frame.to_ndarray(
            format="bgr24"
        )

        detections = self.detect_array(
            frame_array
        )

        if ENABLE_ADAPTIVE_TILING:

            height, width = (
                frame_array.shape[:2]
            )

            step = max(
                1,
                int(
                    TILE_SIZE
                    * (1 - TILE_OVERLAP)
                )
            )

            tile_count = 0

            for y in range(
                0,
                height,
                step
            ):

                for x in range(
                    0,
                    width,
                    step
                ):

                    if tile_count >= MAX_TILES:
                        break

                    tile = frame_array[
                        y:min(y + TILE_SIZE, height),
                        x:min(x + TILE_SIZE, width)
                    ]

                    if (
                        tile.shape[0] < 100
                        or tile.shape[1] < 100
                    ):
                        continue

                    detections.extend(
                        self.detect_array(
                            tile,
                            offset=(x, y)
                        )
                    )

                    tile_count += 1

                if tile_count >= MAX_TILES:
                    break

        return non_max_suppression(
            detections
        )


# ==========================================
# 5. KALMAN TRACKING UTILITIES
# ==========================================

def box_to_measurement(box):

    x1, y1, x2, y2 = box

    return np.array([
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        max(1, x2 - x1),
        max(1, y2 - y1)
    ], dtype=np.float64)


def measurement_to_box(values):

    cx, cy, width, height = values[:4]

    width = max(1, width)
    height = max(1, height)

    return np.array([
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2
    ])


# ==========================================
# 6. KALMAN TRACK
# ==========================================

class FaceTrack:

    def __init__(
        self,
        track_id,
        box,
        confidence,
        frame_number
    ):

        self.id = track_id

        self.state = np.zeros(
            8,
            dtype=np.float64
        )

        self.state[:4] = (
            box_to_measurement(box)
        )

        self.P = np.diag([
            100, 100, 100, 100,
            400, 400, 100, 100
        ]).astype(np.float64)

        self.F = np.eye(8)

        for i in range(4):
            self.F[i, i + 4] = 1

        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        self.Q = np.diag([
            2, 2, 2, 2,
            4, 4, 2, 2
        ]).astype(np.float64)

        self.R = np.diag([
            25, 25, 36, 36
        ]).astype(np.float64)

        self.hits = 1
        self.missed = 0

        self.confidence = confidence
        self.last_detection_frame = (
            frame_number
        )

        self.confirmed = False
        self.best_frames = []

    def predict(self):

        self.state = (
            self.F @ self.state
        )

        self.P = (
            self.F @ self.P @ self.F.T
            + self.Q
        )

        self.state[2:4] = np.maximum(
            self.state[2:4],
            1
        )

        self.missed += 1

    def correct(
        self,
        box,
        confidence,
        frame_number
    ):

        measurement = (
            box_to_measurement(box)
        )

        innovation = (
            measurement
            - self.H @ self.state
        )

        S = (
            self.H @ self.P @ self.H.T
            + self.R
        )

        K = np.linalg.solve(
            S,
            self.H @ self.P
        ).T

        self.state += (
            K @ innovation
        )

        A = (
            np.eye(8)
            - K @ self.H
        )

        self.P = (
            A @ self.P @ A.T
            + K @ self.R @ K.T
        )

        self.state[2:4] = np.maximum(
            self.state[2:4],
            1
        )

        self.missed = 0
        self.hits += 1

        self.confidence = confidence
        self.last_detection_frame = (
            frame_number
        )

        if self.hits >= MIN_TRACK_HITS:
            self.confirmed = True

    def box(self):

        return measurement_to_box(
            self.state
        )

    def gating_distance(self, box):

        measurement = (
            box_to_measurement(box)
        )

        innovation = (
            measurement
            - self.H @ self.state
        )

        S = (
            self.H @ self.P @ self.H.T
            + self.R
        )

        try:

            result = np.linalg.solve(
                S,
                innovation
            )

            return float(
                innovation @ result
            )

        except np.linalg.LinAlgError:
            return float("inf")


# ==========================================
# 7. MULTI-FACE TRACKER
# ==========================================

class MultiFaceTracker:

    def __init__(self):

        self.tracks = []
        self.all_tracks = []

        self.next_id = 1

        self.matches = 0
        self.recoveries = 0
        self.expired = 0

    def add(
        self,
        detection,
        frame_number
    ):

        box, confidence = detection

        track = FaceTrack(
            self.next_id,
            box,
            confidence,
            frame_number
        )

        self.tracks.append(track)
        self.all_tracks.append(track)

        self.next_id += 1

    def associate(
        self,
        track_indices,
        detections,
        max_cost
    ):

        if (
            not track_indices
            or not detections
        ):

            return (
                [],
                track_indices,
                list(range(len(detections)))
            )

        costs = np.full(
            (
                len(track_indices),
                len(detections)
            ),
            1e6
        )

        for row, index in enumerate(
            track_indices
        ):

            track = self.tracks[index]

            predicted_box = track.box()

            predicted_measurement = (
                box_to_measurement(
                    predicted_box
                )
            )

            for col, detection in enumerate(
                detections
            ):

                box, confidence = detection

                measurement = (
                    box_to_measurement(box)
                )

                distance = np.hypot(
                    predicted_measurement[0]
                    - measurement[0],
                    predicted_measurement[1]
                    - measurement[1]
                )

                normalizer = max(
                    1,
                    np.hypot(
                        predicted_measurement[2],
                        predicted_measurement[3]
                    )
                )

                distance /= normalizer

                ratio = max(
                    predicted_measurement[2]
                    / measurement[2],

                    measurement[2]
                    / predicted_measurement[2],

                    predicted_measurement[3]
                    / measurement[3],

                    measurement[3]
                    / predicted_measurement[3]
                )

                overlap = calculate_iou(
                    predicted_box,
                    box
                )

                if (
                    distance > 1.15
                    or ratio > 2.3
                ):
                    continue

                if (
                    overlap < 0.03
                    and distance > 0.45
                ):
                    continue

                gate = (
                    36
                    if track.missed > 2
                    else 20
                )

                if (
                    track.gating_distance(box)
                    > gate
                ):
                    continue

                costs[row, col] = (
                    0.55 * (1 - overlap)
                    + 0.35 * min(
                        1,
                        distance
                    )
                    + 0.10 * min(
                        1,
                        abs(np.log(ratio))
                    )
                )

        rows, cols = (
            linear_sum_assignment(costs)
        )

        matches = []

        used_tracks = set()
        used_detections = set()

        for row, col in zip(
            rows,
            cols
        ):

            if costs[row, col] > max_cost:
                continue

            track_index = (
                track_indices[row]
            )

            matches.append((
                track_index,
                col
            ))

            used_tracks.add(
                track_index
            )

            used_detections.add(
                col
            )

        unmatched_tracks = [
            index
            for index in track_indices
            if index not in used_tracks
        ]

        unmatched_detections = [
            index
            for index in range(
                len(detections)
            )
            if index not in used_detections
        ]

        return (
            matches,
            unmatched_tracks,
            unmatched_detections
        )

    def apply_match(
        self,
        track_index,
        detection,
        frame_number
    ):

        track = (
            self.tracks[track_index]
        )

        if (
            track.missed
            > DETECTION_INTERVAL
        ):
            self.recoveries += 1

        track.correct(
            *detection,
            frame_number
        )

        self.matches += 1

    def update(
        self,
        detections,
        frame_number
    ):

        for track in self.tracks:
            track.predict()

        if detections is not None:

            high = [
                detection
                for detection in detections
                if detection[1] >= HIGH_CONF
            ]

            low = [
                detection
                for detection in detections
                if (
                    LOW_CONF
                    <= detection[1]
                    < HIGH_CONF
                )
            ]

            confirmed = [
                i
                for i, track
                in enumerate(self.tracks)
                if track.confirmed
            ]

            tentative = [
                i
                for i, track
                in enumerate(self.tracks)
                if not track.confirmed
            ]

            matches_1, unmatched, unused_high = (
                self.associate(
                    confirmed,
                    high,
                    0.82
                )
            )

            remaining_high = [
                high[i]
                for i in unused_high
            ]

            matches_2, _, unused_remaining = (
                self.associate(
                    tentative,
                    remaining_high,
                    0.72
                )
            )

            for track_index, detection_index in matches_1:

                self.apply_match(
                    track_index,
                    high[detection_index],
                    frame_number
                )

            for track_index, detection_index in matches_2:

                self.apply_match(
                    track_index,
                    remaining_high[
                        detection_index
                    ],
                    frame_number
                )

            matches_3, _, _ = (
                self.associate(
                    unmatched,
                    low,
                    0.70
                )
            )

            for track_index, detection_index in matches_3:

                self.apply_match(
                    track_index,
                    low[detection_index],
                    frame_number
                )

            for detection_index in unused_remaining:

                self.add(
                    remaining_high[
                        detection_index
                    ],
                    frame_number
                )

        alive = []

        for track in self.tracks:

            limit = (
                MAX_LOST_FRAMES
                if track.confirmed
                else (
                    2 * DETECTION_INTERVAL
                    + 1
                )
            )

            if track.missed <= limit:
                alive.append(track)

            else:
                self.expired += 1

        self.tracks = alive

        visible = [
            track
            for track in self.tracks
            if (
                track.confirmed
                and track.missed
                <= MAX_VISIBLE_MISSED
            )
        ]

        return visible


# ==========================================
# 8. FACE QUALITY
# ==========================================

def assess_face_quality(
    image,
    box,
    occlusion=0
):

    height, width = (
        image.shape[:2]
    )

    x1, y1, x2, y2 = [
        int(round(value))
        for value in box
    ]

    x1 = max(
        0,
        min(width - 1, x1)
    )

    y1 = max(
        0,
        min(height - 1, y1)
    )

    x2 = max(
        0,
        min(width, x2)
    )

    y2 = max(
        0,
        min(height, y2)
    )

    if (
        x2 - x1 < 3
        or y2 - y1 < 3
    ):
        return None

    crop = image[
        y1:y2,
        x1:x2
    ]

    gray = (
        crop.astype(np.float32)
        @ np.array([
            0.299,
            0.587,
            0.114
        ])
    )

    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4 * gray[1:-1, 1:-1]
    )

    blur = float(
        np.var(laplacian)
    )

    brightness = float(
        gray.mean()
    )

    size = min(
        crop.shape[:2]
    )

    eligible = (
        size >= MIN_FACE_SIZE
        and blur >= MIN_BLUR_VARIANCE
        and MIN_BRIGHTNESS
        <= brightness
        <= MAX_BRIGHTNESS
        and occlusion < 0.35
    )

    score = (
        0.35 * min(
            1,
            size / 120
        )

        + 0.40 * min(
            1,
            blur / 250
        )

        + 0.25 * max(
            0,
            1 - abs(
                brightness - 128
            ) / 128
        )
    )

    score *= (
        1 - min(
            1,
            occlusion
        )
    )

    return {
        "eligible": eligible,
        "score": float(score),
        "blur": blur,
        "brightness": brightness,
        "size": size,
        "crop": crop.copy()
    }


# ==========================================
# 9. BEST FRAME SELECTION
# ==========================================

def update_best_frames(
    track,
    quality,
    frame_number
):

    if (
        quality is None
        or not quality["eligible"]
    ):
        return

    nearby = [
        item
        for item in track.best_frames
        if abs(
            item["frame"]
            - frame_number
        ) < MIN_BEST_FRAME_GAP
    ]

    if nearby:

        best_nearby = max(
            item["score"]
            for item in nearby
        )

        if (
            best_nearby
            >= quality["score"]
        ):
            return

        track.best_frames = [
            item
            for item in track.best_frames
            if item not in nearby
        ]

    track.best_frames.append({
        "frame": frame_number,
        "score": quality["score"],
        "crop": quality["crop"]
    })

    track.best_frames.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    track.best_frames = (
        track.best_frames[
            :BEST_FRAMES_PER_TRACK
        ]
    )


# ==========================================
# 10. LIVE PREVIEW
# ==========================================

class LivePreview:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title(
            "SCRFD LIVE PREVIEW - NO FRAME SKIPPING"
        )

        self.root.configure(
            bg="#171717"
        )

        self.image_label = tk.Label(
            self.root,
            bg="black"
        )

        self.image_label.pack(
            padx=8,
            pady=8
        )

        self.status = tk.Label(
            self.root,
            text="Starting...",
            fg="white",
            bg="#171717",
            font=("Arial", 11)
        )

        self.status.pack(
            pady=5
        )

        # Unbounded queue preserves every
        # processed preview frame.
        self.frames = queue.Queue()

        self.stop_event = (
            threading.Event()
        )

        self.photo = None

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

        self.worker = threading.Thread(
            target=self.process,
            daemon=True
        )

        self.worker.start()

        self.root.after(
            10,
            self.refresh
        )

    def put(
        self,
        image,
        message
    ):

        self.frames.put((
            image,
            message
        ))

    def process(self):

        try:

            detector = SCRFDDetector()

            OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            run_number = 0

            while not self.stop_event.is_set():

                run_number += 1

                tracker = (
                    MultiFaceTracker()
                )

                source = av.open(
                    str(VIDEO_PATH)
                )

                stream = (
                    source.streams.video[0]
                )

                fps = float(
                    stream.average_rate
                    or 24
                )

                interval = (
                    1.0 / fps
                )

                counters = defaultdict(int)
                durations = defaultdict(float)

                tracking_rows = []

                video_out = None
                output_stream = None

                if SAVE_ANNOTATED_VIDEO:

                    video_out = av.open(
                        str(
                            OUTPUT_DIR
                            / f"preview_run_{run_number}.mp4"
                        ),
                        mode="w"
                    )

                    output_stream = (
                        video_out.add_stream(
                            "libx264",
                            rate=(
                                stream.average_rate
                                or Fraction(24, 1)
                            ),
                            options={
                                "preset": "ultrafast",
                                "crf": "23"
                            }
                        )
                    )

                    output_stream.width = (
                        stream.width
                    )

                    output_stream.height = (
                        stream.height
                    )

                    output_stream.pix_fmt = (
                        "yuv420p"
                    )

                start = (
                    time.perf_counter()
                )

                try:

                    for frame_number, frame in enumerate(
                        source.decode(video=0)
                    ):

                        if self.stop_event.is_set():
                            break

                        # ==================================
                        # NO FRAME SKIPPING
                        # ==================================

                        target = (
                            start
                            + frame_number * interval
                        )

                        wait = (
                            target
                            - time.perf_counter()
                        )

                        if wait > 0:
                            time.sleep(wait)

                        # No lag-based continue.
                        # Every decoded frame proceeds.

                        frame_start = (
                            time.perf_counter()
                        )

                        detections = None

                        if (
                            frame_number
                            % DETECTION_INTERVAL == 0
                        ):

                            detections = (
                                detector.detect(frame)
                            )

                            counters[
                                "detection_frames"
                            ] += 1

                        detection_end = (
                            time.perf_counter()
                        )

                        tracks = tracker.update(
                            detections,
                            frame_number
                        )

                        tracking_end = (
                            time.perf_counter()
                        )

                        rgb = frame.to_ndarray(
                            format="rgb24"
                        )

                        # ==================================
                        # QUALITY CHECKS
                        # ==================================

                        for track in tracks:

                            box = track.box()

                            if (
                                track.last_detection_frame
                                == frame_number
                            ):

                                occlusion = max(
                                    (
                                        calculate_iou(
                                            box,
                                            other.box()
                                        )
                                        for other in tracks
                                        if other is not track
                                    ),
                                    default=0
                                )

                                quality = (
                                    assess_face_quality(
                                        rgb,
                                        box,
                                        occlusion
                                    )
                                )

                                if quality:

                                    counters[
                                        "quality_checked"
                                    ] += 1

                                    if quality["eligible"]:

                                        counters[
                                            "quality_eligible"
                                        ] += 1

                                    update_best_frames(
                                        track,
                                        quality,
                                        frame_number
                                    )

                            tracking_rows.append([
                                frame_number,
                                round(
                                    frame_number / fps,
                                    3
                                ),
                                track.id,
                                *[
                                    round(
                                        float(value),
                                        2
                                    )
                                    for value in box
                                ],
                                round(
                                    track.confidence,
                                    3
                                ),
                                track.missed
                            ])

                        # ==================================
                        # DRAW BOXES
                        # ==================================

                        image = Image.fromarray(
                            rgb
                        )

                        draw = ImageDraw.Draw(
                            image
                        )

                        for track in tracks:

                            x1, y1, x2, y2 = [
                                int(value)
                                for value in track.box()
                            ]

                            x1 = max(0, x1)
                            y1 = max(0, y1)

                            x2 = min(
                                image.width - 1,
                                x2
                            )

                            y2 = min(
                                image.height - 1,
                                y2
                            )

                            if (
                                x2 <= x1
                                or y2 <= y1
                            ):
                                continue

                            draw.rectangle(
                                (
                                    x1,
                                    y1,
                                    x2,
                                    y2
                                ),
                                outline="#00ff66",
                                width=3
                            )

                            draw.text(
                                (
                                    x1,
                                    max(0, y1 - 15)
                                ),
                                f"ID {track.id}",
                                fill="#00ff66",
                                stroke_width=1,
                                stroke_fill="black"
                            )

                        counters[
                            "processed_frames"
                        ] += 1

                        counters[
                            "drawn_boxes"
                        ] += len(tracks)

                        counters[
                            "maximum_visible_tracks"
                        ] = max(
                            counters[
                                "maximum_visible_tracks"
                            ],
                            len(tracks)
                        )

                        # ==================================
                        # OPTIONAL RECORDING
                        # ==================================

                        if video_out is not None:

                            output_frame = (
                                av.VideoFrame.from_ndarray(
                                    np.asarray(image),
                                    format="rgb24"
                                )
                            )

                            for packet in (
                                output_stream.encode(
                                    output_frame
                                )
                            ):
                                video_out.mux(
                                    packet
                                )

                        frame_end = (
                            time.perf_counter()
                        )

                        durations[
                            "detection"
                        ] += (
                            detection_end
                            - frame_start
                        )

                        durations[
                            "tracking"
                        ] += (
                            tracking_end
                            - detection_end
                        )

                        durations[
                            "quality_draw_encode"
                        ] += (
                            frame_end
                            - tracking_end
                        )

                        preview = image.copy()

                        preview.thumbnail((
                            PREVIEW_MAX_WIDTH,
                            1000
                        ))

                        elapsed_frame = (
                            frame_end
                            - frame_start
                        )

                        message = (
                            f"Input FPS: {fps:.1f} | "
                            f"Frame time: "
                            f"{elapsed_frame * 1000:.1f} ms | "
                            f"Tracks: {len(tracks)} | "
                            f"IDs: {tracker.next_id - 1} | "
                            f"Processed: "
                            f"{counters['processed_frames']} | "
                            f"Skipped: 0"
                        )

                        self.put(
                            preview,
                            message
                        )

                        if (
                            counters[
                                "processed_frames"
                            ] % 100 == 0
                        ):

                            print(
                                "Processed",
                                counters[
                                    "processed_frames"
                                ],
                                "frames | Skipped: 0"
                            )

                    if video_out is not None:

                        for packet in (
                            output_stream.encode()
                        ):
                            video_out.mux(
                                packet
                            )

                finally:

                    source.close()

                    if video_out is not None:
                        video_out.close()

                total_time = (
                    time.perf_counter()
                    - start
                )

                # ==================================
                # SAVE BEST FACE IMAGES
                # ==================================

                best_directory = (
                    OUTPUT_DIR
                    / f"best_faces_run_{run_number}"
                )

                best_directory.mkdir(
                    parents=True,
                    exist_ok=True
                )

                saved_faces = 0

                for track in tracker.all_tracks:

                    if not track.confirmed:
                        continue

                    for rank, item in enumerate(
                        track.best_frames,
                        start=1
                    ):

                        filename = (
                            best_directory
                            / (
                                f"id_{track.id:03d}_"
                                f"rank_{rank}_"
                                f"frame_{item['frame']}.jpg"
                            )
                        )

                        Image.fromarray(
                            item["crop"]
                        ).save(
                            filename,
                            quality=92
                        )

                        saved_faces += 1

                # ==================================
                # SAVE TRACKING CSV
                # ==================================

                csv_path = (
                    OUTPUT_DIR
                    / f"tracks_run_{run_number}.csv"
                )

                with csv_path.open(
                    "w",
                    newline="",
                    encoding="utf-8"
                ) as file:

                    writer = csv.writer(
                        file
                    )

                    writer.writerow([
                        "frame",
                        "time_seconds",
                        "track_id",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "confidence",
                        "missed_frames"
                    ])

                    writer.writerows(
                        tracking_rows
                    )

                # ==================================
                # SAVE REPORT
                # ==================================

                processed = counters[
                    "processed_frames"
                ]

                report = {
                    "run": run_number,
                    "input_fps": fps,
                    "processed_frames": processed,
                    "skipped_frames": 0,
                    "total_time_seconds": round(
                        total_time,
                        3
                    ),
                    "effective_processing_fps": round(
                        processed
                        / max(total_time, 1e-9),
                        2
                    ),
                    "detection_runs": counters[
                        "detection_frames"
                    ],
                    "track_ids_created": (
                        tracker.next_id - 1
                    ),
                    "confirmed_tracks": sum(
                        track.confirmed
                        for track in tracker.all_tracks
                    ),
                    "successful_matches": (
                        tracker.matches
                    ),
                    "recoveries": (
                        tracker.recoveries
                    ),
                    "expired_tracks": (
                        tracker.expired
                    ),
                    "maximum_visible_tracks": (
                        counters[
                            "maximum_visible_tracks"
                        ]
                    ),
                    "quality_checked": counters[
                        "quality_checked"
                    ],
                    "quality_eligible": counters[
                        "quality_eligible"
                    ],
                    "best_face_images_saved": (
                        saved_faces
                    ),
                    "average_ms_per_frame": {
                        key: round(
                            value * 1000
                            / max(processed, 1),
                            2
                        )
                        for key, value
                        in durations.items()
                    },
                    "notes": [
                        "No intentional frame skipping",
                        "Every decoded frame tracked",
                        "SCRFD detection every configured interval",
                        "CPU only",
                        "No OpenCV",
                        "No face recognition",
                        "Playback may be slower than real time"
                    ]
                }

                report_path = (
                    OUTPUT_DIR
                    / f"report_run_{run_number}.json"
                )

                report_path.write_text(
                    json.dumps(
                        report,
                        indent=2
                    ),
                    encoding="utf-8"
                )

                print()
                print(
                    "========== SCRFD RESULTS =========="
                )

                print(
                    "Processed frames:",
                    processed
                )

                print(
                    "Skipped frames: 0"
                )

                print(
                    "Detection runs:",
                    counters[
                        "detection_frames"
                    ]
                )

                print(
                    "Track IDs created:",
                    tracker.next_id - 1
                )

                print(
                    "Successful matches:",
                    tracker.matches
                )

                print(
                    "Total time:",
                    round(total_time, 2),
                    "seconds"
                )

                print(
                    "Effective FPS:",
                    report[
                        "effective_processing_fps"
                    ]
                )

                print(
                    "Output directory:",
                    OUTPUT_DIR
                )

                print(
                    "==================================="
                )

                if not LOOP_VIDEO:

                    self.put(
                        None,
                        (
                            "Video completed. "
                            f"Processed {processed} frames. "
                            "Skipped: 0. "
                            "Reports saved."
                        )
                    )

                    break

        except Exception as error:

            self.put(
                None,
                (
                    f"ERROR: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )
            )

            print(
                "ERROR:",
                repr(error)
            )

    def refresh(self):

        try:

            # Display queued frames in order.
            # No preview frames discarded.
            image, message = (
                self.frames.get_nowait()
            )

            if image is not None:

                self.photo = (
                    ImageTk.PhotoImage(
                        image
                    )
                )

                self.image_label.configure(
                    image=self.photo
                )

            self.status.configure(
                text=message
            )

        except queue.Empty:
            pass

        if not self.stop_event.is_set():

            self.root.after(
                10,
                self.refresh
            )

    def close(self):

        self.stop_event.set()

        self.root.destroy()

    def run(self):

        self.root.mainloop()


# ==========================================
# 11. MAIN
# ==========================================

if __name__ == "__main__":

    if not VIDEO_PATH.is_file():

        raise FileNotFoundError(
            f"Input video not found: "
            f"{VIDEO_PATH}"
        )

    print(
        "SCRFD LIVE PREVIEW - NO FRAME SKIPPING"
    )

    print(
        "Video:",
        VIDEO_PATH
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Detection interval:",
        DETECTION_INTERVAL
    )

    

    LivePreview().run()
