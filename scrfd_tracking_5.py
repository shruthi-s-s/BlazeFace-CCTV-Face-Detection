from pathlib import Path
from fractions import Fraction
import csv
import json
import time

import av
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.optimize import linear_sum_assignment

from scrfd_detect import decode_outputs, non_max_suppression


# ==========================================
# CONFIGURATION
# ==========================================

BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "models" / "2.5g_bnkps.onnx"
VIDEO_PATH = BASE / "videos" / "input.mp4"
OUTPUT_PATH = BASE / "videos" / "scrfd_tracking_5.mp4"

CSV_PATH = BASE / "videos" / "scrfd_tracks.csv"
REPORT_PATH = BASE / "videos" / "scrfd_tracking_report.json"
BEST_DIR = BASE / "videos" / "best_faces"

MODEL_SIZE = 640
DETECTION_INTERVAL = 2
CONFIDENCE_THRESHOLD = 0.55

# Tracking configuration
MIN_CONFIRMATION_HITS = 3
MAX_LOST_FRAMES = 36
MAX_VISIBLE_MISSED = 3

MAX_CENTER_DISTANCE = 1.15
MIN_IOU = 0.08
MAX_SIZE_RATIO = 2.0

MAX_MATCH_COST = 0.85
MATCH_MAHALANOBIS_LIMIT = 16.0

# Quality assessment
MIN_FACE_SIZE = 48
MIN_BLUR_VARIANCE = 55.0
MIN_BRIGHTNESS = 45.0
MAX_BRIGHTNESS = 215.0

BEST_FRAMES_PER_TRACK = 5
MIN_BEST_FRAME_GAP = 5

# Keep advanced processing disabled while
# measuring tracking stability.
ENABLE_ADAPTIVE_TILING = False
ENABLE_CLAHE = False

BOX_THICKNESS = 3


# ==========================================
# BOX UTILITIES
# ==========================================

def box_to_measurement(box):
    x1, y1, x2, y2 = map(float, box)

    return np.array([
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        max(1.0, x2 - x1),
        max(1.0, y2 - y1)
    ], dtype=np.float64)


def measurement_to_box(values):
    cx, cy, w, h = values[:4]

    w = max(1.0, float(w))
    h = max(1.0, float(h))

    return np.array([
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2
    ], dtype=np.float64)


def calculate_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area_a = (
        max(0, a[2] - a[0]) *
        max(0, a[3] - a[1])
    )

    area_b = (
        max(0, b[2] - b[0]) *
        max(0, b[3] - b[1])
    )

    return intersection / max(
        area_a + area_b - intersection,
        1e-6
    )


def center_distance(a, b):
    ma = box_to_measurement(a)
    mb = box_to_measurement(b)

    distance = np.hypot(
        ma[0] - mb[0],
        ma[1] - mb[1]
    )

    normalizer = max(
        1.0,
        np.hypot(
            (ma[2] + mb[2]) / 2,
            (ma[3] + mb[3]) / 2
        )
    )

    return float(distance / normalizer)


def size_ratio(a, b):
    ma = box_to_measurement(a)
    mb = box_to_measurement(b)

    return max(
        ma[2] / mb[2],
        mb[2] / ma[2],
        ma[3] / mb[3],
        mb[3] / ma[3]
    )


# ==========================================
# KALMAN FACE TRACK
# ==========================================

class FaceTrack:

    def __init__(
        self,
        track_id,
        box,
        confidence,
        frame_index
    ):
        self.track_id = track_id

        # cx, cy, width, height,
        # vx, vy, vw, vh
        self.state = np.zeros(
            8, dtype=np.float64
        )

        self.state[:4] = box_to_measurement(box)

        self.P = np.diag([
            100, 100, 100, 100,
            400, 400, 100, 100
        ]).astype(np.float64)

        self.F = np.eye(8)

        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        self.Q = np.diag([
            1, 1, 1, 1,
            4, 4, 1, 1
        ]).astype(np.float64)

        self.R = np.diag([
            16, 16, 25, 25
        ]).astype(np.float64)

        self.hits = 1
        self.missed = 0

        self.confidence = float(confidence)
        self.last_detection_frame = frame_index

        self.confirmed = False
        self.best_frames = []

    def predict(self):
        self.state = self.F @ self.state

        self.P = (
            self.F @ self.P @ self.F.T +
            self.Q
        )

        self.state[2:4] = np.maximum(
            self.state[2:4], 1.0
        )

        self.missed += 1

    def correct(
        self,
        box,
        confidence,
        frame_index
    ):
        z = box_to_measurement(box)

        innovation = z - self.H @ self.state

        S = self.H @ self.P @ self.H.T + self.R

        K = np.linalg.solve(
            S,
            self.H @ self.P
        ).T

        self.state = (
            self.state + K @ innovation
        )

        A = np.eye(8) - K @ self.H

        self.P = (
            A @ self.P @ A.T +
            K @ self.R @ K.T
        )

        self.state[2:4] = np.maximum(
            self.state[2:4], 1.0
        )

        self.hits += 1
        self.missed = 0

        self.confidence = float(confidence)
        self.last_detection_frame = frame_index

        if self.hits >= MIN_CONFIRMATION_HITS:
            self.confirmed = True

    def box(self):
        return measurement_to_box(self.state)

    def mahalanobis_distance(self, box):
        z = box_to_measurement(box)

        difference = (
            z - self.H @ self.state
        )

        S = self.H @ self.P @ self.H.T + self.R

        try:
            result = np.linalg.solve(
                S, difference
            )

            return float(
                difference @ result
            )

        except np.linalg.LinAlgError:
            return float("inf")


# ==========================================
# HUNGARIAN MULTI-FACE TRACKER
# ==========================================

class MultiFaceTracker:

    def __init__(self):
        self.tracks = []
        self.all_tracks = []

        self.next_id = 1

        self.created = 0
        self.matches = 0
        self.expired = 0

        self.confirmed_created = 0
        self.recovered = 0

    def add(self, box, confidence, frame_index):
        track = FaceTrack(
            self.next_id,
            box,
            confidence,
            frame_index
        )

        self.tracks.append(track)
        self.all_tracks.append(track)

        self.next_id += 1
        self.created += 1

    def association_cost(self, track, detection):
        box, confidence = detection

        predicted = track.box()

        iou = calculate_iou(
            predicted, box
        )

        distance = center_distance(
            predicted, box
        )

        ratio = size_ratio(
            predicted, box
        )

        mahalanobis = track.mahalanobis_distance(
            box
        )

        if distance > MAX_CENTER_DISTANCE:
            return None

        if ratio > MAX_SIZE_RATIO:
            return None

        if mahalanobis > MATCH_MAHALANOBIS_LIMIT:
            return None

        # Require overlap or close centers.
        if iou < MIN_IOU and distance > 0.45:
            return None

        size_penalty = min(
            1.0,
            abs(np.log(ratio))
        )

        cost = (
            0.50 * (1.0 - iou) +
            0.35 * min(distance, 1.0) +
            0.15 * size_penalty
        )

        return float(cost)

    def update(self, detections, frame_index):

        # Predict every track on every frame.
        for track in self.tracks:
            track.predict()

        if detections is not None:
            detections = list(detections)

            track_count = len(self.tracks)
            detection_count = len(detections)

            matched_tracks = set()
            matched_detections = set()

            if track_count and detection_count:
                # Large cost means forbidden match.
                cost_matrix = np.full(
                    (track_count, detection_count),
                    1e6,
                    dtype=np.float64
                )

                for ti, track in enumerate(self.tracks):
                    for di, detection in enumerate(detections):
                        cost = self.association_cost(
                            track, detection
                        )

                        if cost is not None:
                            cost_matrix[ti, di] = cost

                # Global minimum-cost assignment.
                row_indices, col_indices = (
                    linear_sum_assignment(cost_matrix)
                )

                for ti, di in zip(
                    row_indices, col_indices
                ):
                    cost = cost_matrix[ti, di]

                    if cost > MAX_MATCH_COST:
                        continue

                    track = self.tracks[ti]
                    box, confidence = detections[di]

                    previous_missed = track.missed

                    was_confirmed = track.confirmed

                    track.correct(
                        box,
                        confidence,
                        frame_index
                    )

                    if (
                        not was_confirmed
                        and track.confirmed
                    ):
                        self.confirmed_created += 1

                    # More than one elapsed frame
                    # indicates a detection gap.
                    if previous_missed > DETECTION_INTERVAL:
                        self.recovered += 1

                    matched_tracks.add(ti)
                    matched_detections.add(di)

                    self.matches += 1

            for di, detection in enumerate(detections):
                if di in matched_detections:
                    continue

                box, confidence = detection

                self.add(
                    box,
                    confidence,
                    frame_index
                )

        retained = []

        for track in self.tracks:

            # Tentative tracks get a shorter
            # recovery window.
            limit = (
                MAX_LOST_FRAMES
                if track.confirmed
                else 2 * DETECTION_INTERVAL + 1
            )

            if track.missed <= limit:
                retained.append(track)
            else:
                self.expired += 1

        self.tracks = retained

        visible = [
            track for track in self.tracks
            if (
                track.confirmed
                and track.missed <= MAX_VISIBLE_MISSED
            )
        ]

        return visible


# ==========================================
# SCRFD PREPROCESSING
# ==========================================

def preprocess_frame(frame):
    width = frame.width
    height = frame.height

    scale = min(
        MODEL_SIZE / width,
        MODEL_SIZE / height
    )

    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))

    resized = frame.reformat(
        width=new_w,
        height=new_h,
        format="bgr24"
    ).to_ndarray()

    canvas = np.zeros(
        (MODEL_SIZE, MODEL_SIZE, 3),
        dtype=np.uint8
    )

    canvas[:new_h, :new_w] = resized

    tensor = (
        canvas.astype(np.float32) - 127.5
    ) / 128.0

    tensor = np.ascontiguousarray(
        tensor.transpose(2, 0, 1)[None, ...]
    )

    return tensor, scale


def detect_faces(session, input_name, frame):
    tensor, scale = preprocess_frame(frame)

    outputs = session.run(
        None,
        {input_name: tensor}
    )

    boxes, scores = decode_outputs(
        outputs, scale
    )

    selected = non_max_suppression(
        boxes, scores
    )

    detections = []

    for i in selected:
        confidence = float(scores[i])

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        box = np.asarray(
            boxes[i],
            dtype=np.float64
        ).reshape(4)

        if not np.all(np.isfinite(box)):
            continue

        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        detections.append(
            (box.copy(), confidence)
        )

    return detections


# ==========================================
# FACE QUALITY ASSESSMENT
# ==========================================

def assess_quality(image, box):
    height, width = image.shape[:2]

    x1, y1, x2, y2 = [
        int(round(float(value)))
        for value in box
    ]

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))

    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]

    if min(crop.shape[:2]) < 3:
        return None

    gray = (
        0.299 * crop[:, :, 0].astype(np.float32) +
        0.587 * crop[:, :, 1].astype(np.float32) +
        0.114 * crop[:, :, 2].astype(np.float32)
    )

    laplacian = (
        gray[:-2, 1:-1] +
        gray[2:, 1:-1] +
        gray[1:-1, :-2] +
        gray[1:-1, 2:] -
        4 * gray[1:-1, 1:-1]
    )

    blur = float(np.var(laplacian))
    brightness = float(np.mean(gray))
    face_size = min(crop.shape[:2])

    eligible = (
        face_size >= MIN_FACE_SIZE
        and blur >= MIN_BLUR_VARIANCE
        and MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS
    )

    quality_score = (
        0.40 * min(1.0, blur / 250.0) +
        0.35 * min(1.0, face_size / 120.0) +
        0.25 * max(
            0.0,
            1.0 - abs(brightness - 128) / 128
        )
    )

    return {
        "eligible": eligible,
        "score": float(quality_score),
        "blur": blur,
        "brightness": brightness,
        "size": face_size,
        "crop": crop.copy()
    }


def update_best_frames(track, quality, frame_index):
    if not quality["eligible"]:
        return

    candidate = {
        "frame": frame_index,
        "score": quality["score"],
        "crop": quality["crop"]
    }

    nearby = [
        item for item in track.best_frames
        if abs(item["frame"] - frame_index)
        < MIN_BEST_FRAME_GAP
    ]

    if nearby:
        if max(
            item["score"] for item in nearby
        ) >= candidate["score"]:
            return

        track.best_frames = [
            item for item in track.best_frames
            if all(
                item is not old
                for old in nearby
            )
        ]

    track.best_frames.append(candidate)

    track.best_frames.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    track.best_frames = track.best_frames[
        :BEST_FRAMES_PER_TRACK
    ]


# ==========================================
# DRAW TRACK BOXES AND IDS
# ==========================================

FONT = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
}


def draw_id(image, x, y, track_id):
    text = str(track_id)

    height, width = image.shape[:2]

    scale = 2
    label_width = len(text) * 8 + 6
    label_height = 16

    if label_width > width or label_height > height:
        return

    x = max(0, min(x, width - label_width))
    y = max(
        0,
        min(y - label_height, height - label_height)
    )

    image[
        y:y + label_height,
        x:x + label_width
    ] = (0, 100, 0)

    for index, char in enumerate(text):
        pattern = FONT.get(char)

        if pattern is None:
            continue

        for row, pixels in enumerate(pattern):
            for col, pixel in enumerate(pixels):
                if pixel == "1":
                    px = x + 3 + index * 8 + col * scale
                    py = y + 3 + row * scale

                    image[
                        py:py + scale,
                        px:px + scale
                    ] = (255, 255, 255)


def draw_tracks(image, tracks):
    height, width = image.shape[:2]

    for track in tracks:
        x1, y1, x2, y2 = [
            int(round(float(v)))
            for v in track.box()
        ]

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))

        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        t = BOX_THICKNESS

        image[
            y1:min(height, y1 + t),
            x1:x2 + 1
        ] = (0, 255, 0)

        image[
            max(0, y2 - t + 1):y2 + 1,
            x1:x2 + 1
        ] = (0, 255, 0)

        image[
            y1:y2 + 1,
            x1:min(width, x1 + t)
        ] = (0, 255, 0)

        image[
            y1:y2 + 1,
            max(0, x2 - t + 1):x2 + 1
        ] = (0, 255, 0)

        draw_id(
            image,
            x1,
            y1,
            track.track_id
        )


# ==========================================
# MAIN VIDEO PROCESSING
# ==========================================

def main():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(VIDEO_PATH)

    options = ort.SessionOptions()

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    tracker = MultiFaceTracker()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    source = av.open(str(VIDEO_PATH))
    video_stream = source.streams.video[0]

    fps = Fraction(
        video_stream.average_rate
        or Fraction(24, 1)
    )

    destination = av.open(
        str(OUTPUT_PATH),
        mode="w"
    )

    output_stream = destination.add_stream(
        "libx264",
        rate=fps,
        options={
            "preset": "ultrafast",
            "crf": "23",
            "threads": "2"
        }
    )

    output_stream.width = video_stream.width
    output_stream.height = video_stream.height
    output_stream.pix_fmt = "yuv420p"

    timings = {
        "detection": 0.0,
        "tracking": 0.0,
        "quality": 0.0,
        "conversion_drawing_encoding": 0.0
    }

    frame_count = 0
    detection_runs = 0
    max_visible_tracks = 0
    rows = []

    print("===================================")
    print("SCRFD - HUNGARIAN KALMAN TRACKING")
    print("===================================")
    print("Input:", VIDEO_PATH.name)
    print("FPS:", float(fps))
    print("Detection interval:", DETECTION_INTERVAL)
    print("Matching: Hungarian + motion gating")
    print("Face recognition: Disabled")
    print("CPU only / No OpenCV")
    print("===================================")

    start_total = time.perf_counter()

    try:
        for frame in source.decode(video=0):

            detections = None

            if frame_count % DETECTION_INTERVAL == 0:
                start = time.perf_counter()

                detections = detect_faces(
                    session,
                    input_name,
                    frame
                )

                timings["detection"] += (
                    time.perf_counter() - start
                )

                detection_runs += 1

            start = time.perf_counter()

            visible = tracker.update(
                detections,
                frame_count
            )

            timings["tracking"] += (
                time.perf_counter() - start
            )

            max_visible_tracks = max(
                max_visible_tracks,
                len(visible)
            )

            start = time.perf_counter()

            image = frame.to_ndarray(
                format="rgb24"
            )

            timings["conversion_drawing_encoding"] += (
                time.perf_counter() - start
            )

            start = time.perf_counter()

            for track in visible:

                # Quality checks only on fresh
                # detections, not predicted boxes.
                if track.last_detection_frame == frame_count:

                    quality = assess_quality(
                        image,
                        track.box()
                    )

                    if quality is not None:
                        update_best_frames(
                            track,
                            quality,
                            frame_count
                        )

                rows.append([
                    frame_count,
                    round(frame_count / float(fps), 3),
                    track.track_id,
                    *[
                        round(float(v), 2)
                        for v in track.box()
                    ],
                    round(track.confidence, 4),
                    track.missed
                ])

            timings["quality"] += (
                time.perf_counter() - start
            )

            start = time.perf_counter()

            draw_tracks(image, visible)

            output_frame = av.VideoFrame.from_ndarray(
                image,
                format="rgb24"
            )

            for packet in output_stream.encode(
                output_frame
            ):
                destination.mux(packet)

            timings["conversion_drawing_encoding"] += (
                time.perf_counter() - start
            )

            frame_count += 1

            if frame_count % 100 == 0:
                print(
                    f"Processed {frame_count} frames"
                )

        for packet in output_stream.encode():
            destination.mux(packet)

    finally:
        destination.close()
        source.close()

    elapsed = time.perf_counter() - start_total

    # ==========================================
    # SAVE BEST FACE CROPS
    # ==========================================

    BEST_DIR.mkdir(
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
            filename = BEST_DIR / (
                f"id_{track.track_id:03d}_"
                f"rank_{rank}_"
                f"frame_{item['frame']}.jpg"
            )

            Image.fromarray(
                item["crop"]
            ).save(filename, quality=92)

            saved_faces += 1

    # ==========================================
    # SAVE CSV
    # ==========================================

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "frame",
            "time_seconds",
            "track_id",
            "x1", "y1", "x2", "y2",
            "confidence",
            "frames_since_detection"
        ])

        writer.writerows(rows)

    # ==========================================
    # SAVE REPORT
    # ==========================================

    report = {
        "frames": frame_count,
        "input_fps": float(fps),
        "detection_runs": detection_runs,
        "processing_seconds": round(elapsed, 3),
        "processing_fps": round(
            frame_count / max(elapsed, 1e-9),
            3
        ),
        "track_ids_created": tracker.created,
        "confirmed_tracks": tracker.confirmed_created,
        "successful_matches": tracker.matches,
        "tracks_recovered": tracker.recovered,
        "tracks_expired": tracker.expired,
        "maximum_visible_tracks": max_visible_tracks,
        "best_face_crops_saved": saved_faces,
        "average_ms_per_frame": {
            name: round(
                value * 1000 / max(frame_count, 1),
                3
            )
            for name, value in timings.items()
        },
        "notes": [
            "Hungarian matching with Kalman motion gating",
            "No face recognition or re-identification",
            "ID switches require ground-truth evaluation",
            "Best-face crops grouped by tracking ID",
            "Offline FPS is not live CCTV FPS"
        ]
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n========== TRACKING RESULTS ==========")

    print("Total frames:", frame_count)
    print("Detection runs:", detection_runs)
    print("Track IDs created:", tracker.created)
    print("Confirmed tracks:", tracker.confirmed_created)
    print("Successful matches:", tracker.matches)
    print("Tracks recovered:", tracker.recovered)
    print("Tracks expired:", tracker.expired)
    print("Maximum visible tracks:", max_visible_tracks)

    print(f"Total time: {elapsed:.2f} s")

    print(
        "Processing FPS:",
        f"{report['processing_fps']:.2f}"
    )

    print("\nAverage timings per frame:")

    for name, value in report["average_ms_per_frame"].items():
        print(f"{name}: {value:.2f} ms")

    print("\nOutput:", OUTPUT_PATH)
    print("CSV:", CSV_PATH)
    print("Report:", REPORT_PATH)
    print("Best faces:", BEST_DIR)

    print("===================================")


if __name__ == "__main__":
    main()
