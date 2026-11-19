
import time
import json
import csv
import queue
import threading
import tkinter as tk

from pathlib import Path

import av
import numpy as np
import onnxruntime as ort

from PIL import Image, ImageDraw, ImageTk
from scipy.optimize import linear_sum_assignment


# ==========================================
# CONFIGURATION
# ==========================================

BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "models" / "det_10g.onnx"
VIDEO_PATH = BASE / "videos" / "input.mp4"
OUTPUT_DIR = BASE / "videos" / "live_results_10g"

MODEL_SIZE = 640

HIGH_CONF = 0.50
LOW_CONF = 0.05

NMS_THRESHOLD = 0.40

USE_RGB_INPUT = True
ENABLE_CLAHE = False
ENABLE_ADAPTIVE_TILING = False

DETECTION_INTERVAL = 2

MIN_TRACK_HITS = 2
MAX_LOST_FRAMES = 30
MAX_VISIBLE_MISSED = 3

MIN_FACE_SIZE = 40
MIN_BLUR_VARIANCE = 45.0

BEST_FRAMES_PER_TRACK = 5
MIN_BEST_FRAME_GAP = 5

PREVIEW_MAX_WIDTH = 1100


# ==========================================
# IOU
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


# ==========================================
# NMS
# ==========================================

def non_max_suppression(
    detections,
    threshold=NMS_THRESHOLD
):

    if not detections:
        return []

    boxes = np.asarray(
        [d[0] for d in detections]
    )

    scores = np.asarray(
        [d[1] for d in detections]
    )

    x1, y1, x2, y2 = boxes.T

    areas = (
        np.maximum(0, x2 - x1)
        * np.maximum(0, y2 - y1)
    )

    order = np.argsort(-scores)

    keep = []

    while order.size:

        i = order[0]

        keep.append(detections[i])

        rest = order[1:]

        if not rest.size:
            break

        intersection = (
            np.maximum(
                0,
                np.minimum(x2[i], x2[rest])
                - np.maximum(x1[i], x1[rest])
            )
            *
            np.maximum(
                0,
                np.minimum(y2[i], y2[rest])
                - np.maximum(y1[i], y1[rest])
            )
        )

        union = (
            areas[i] + areas[rest]
            - intersection
        )

        iou = intersection / np.maximum(
            union, 1e-6
        )

        order = rest[iou <= threshold]

    return keep


# ==========================================
# SCRFD 10G DETECTOR
# ==========================================

class SCRFDDetector:

    def __init__(
        self,
        model_path=MODEL_PATH
    ):

        if not Path(model_path).exists():

            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        options = ort.SessionOptions()

        options.graph_optimization_level = (
            ort.GraphOptimizationLevel
            .ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = (
            self.session.get_inputs()[0].name
        )

        self.output_names = [
            out.name
            for out in self.session.get_outputs()
        ]

        self.anchor_cache = {}

        self.inference_count = 0
        self.inference_time = 0.0

        print("\nSCRFD detector initialized")
        print("Model:", model_path)
        print("Provider:", self.session.get_providers())
        print("Input:", self.input_name)

        for output in self.session.get_outputs():

            print(
                "Output:",
                output.name,
                output.shape
            )

    def preprocess(
        self,
        rgb,
        rgb_input=True,
        clahe=False
    ):

        height, width = rgb.shape[:2]

        scale = min(
            MODEL_SIZE / width,
            MODEL_SIZE / height
        )

        new_width = max(
            1, int(width * scale)
        )

        new_height = max(
            1, int(height * scale)
        )

        if clahe:

            from skimage.exposure import equalize_adapthist

            rgb = np.uint8(
                np.clip(
                    equalize_adapthist(
                        rgb,
                        clip_limit=0.02
                    ) * 255,
                    0,
                    255
                )
            )

        resized = np.asarray(
            Image.fromarray(rgb).resize(
                (new_width, new_height),
                Image.Resampling.BILINEAR
            )
        )

        if not rgb_input:

            resized = resized[:, :, ::-1]

        canvas = np.zeros(
            (MODEL_SIZE, MODEL_SIZE, 3),
            dtype=np.uint8
        )

        canvas[
            :new_height,
            :new_width
        ] = resized

        tensor = (
            canvas.astype(np.float32)
            - 127.5
        ) / 128.0

        tensor = np.ascontiguousarray(
            tensor.transpose(2, 0, 1)[None]
        )

        return (
            tensor,
            new_width / width,
            new_height / height
        )

    def decode_outputs(self, raw_outputs):

        outputs = {}

        for output in raw_outputs:

            arr = np.asarray(output)

            if arr.ndim < 2:
                continue

            channels = arr.shape[-1]

            if channels not in (1, 4, 10):
                continue

            arr = arr.reshape(-1, channels)

            for stride in (8, 16, 32):

                expected = (
                    2 * (MODEL_SIZE // stride) ** 2
                )

                if len(arr) != expected:
                    continue

                prefix = {
                    1: "score",
                    4: "bbox",
                    10: "kps"
                }[channels]

                outputs[
                    f"{prefix}_{stride}"
                ] = arr

        for stride in (8, 16, 32):

            for prefix in ("score", "bbox"):

                key = f"{prefix}_{stride}"

                if key not in outputs:

                    raise ValueError(
                        f"Missing output: {key}. "
                        f"Model shapes: "
                        f"{[o.shape for o in raw_outputs]}"
                    )

        return outputs

    def get_anchors(self, stride, count):

        key = (stride, count)

        if key in self.anchor_cache:

            return self.anchor_cache[key]

        grid = MODEL_SIZE // stride

        yy, xx = np.mgrid[:grid, :grid]

        centers = np.stack(
            (xx, yy),
            axis=-1
        ).reshape(-1, 2)

        if count % len(centers):

            raise ValueError(
                "Invalid anchor count"
            )

        centers = np.repeat(
            centers,
            count // len(centers),
            axis=0
        ).astype(np.float32)

        centers *= stride

        self.anchor_cache[key] = centers

        return centers

    def detect_raw(
        self,
        rgb,
        rgb_input=USE_RGB_INPUT,
        clahe=ENABLE_CLAHE,
        minimum_confidence=LOW_CONF
    ):

        height, width = rgb.shape[:2]

        tensor, sx, sy = self.preprocess(
            rgb,
            rgb_input,
            clahe
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

        outputs = self.decode_outputs(
            raw_outputs
        )

        detections = []

        for stride in (8, 16, 32):

            scores = outputs[
                f"score_{stride}"
            ].reshape(-1)

            distances = outputs[
                f"bbox_{stride}"
            ].reshape(-1, 4)

            anchors = self.get_anchors(
                stride,
                len(scores)
            )

            selected = np.where(
                np.isfinite(scores)
                & (scores >= minimum_confidence)
                & (scores <= 1)
            )[0]

            if not selected.size:
                continue

            c = anchors[selected]

            d = (
                distances[selected] * stride
            )

            boxes = np.column_stack([
                c[:, 0] - d[:, 0],
                c[:, 1] - d[:, 1],
                c[:, 0] + d[:, 2],
                c[:, 1] + d[:, 3]
            ])

            boxes[:, [0, 2]] /= sx
            boxes[:, [1, 3]] /= sy

            boxes[:, [0, 2]] = np.clip(
                boxes[:, [0, 2]],
                0,
                width
            )

            boxes[:, [1, 3]] = np.clip(
                boxes[:, [1, 3]],
                0,
                height
            )

            for box, score in zip(
                boxes,
                scores[selected]
            ):

                if not np.all(np.isfinite(box)):
                    continue

                if (
                    box[2] <= box[0]
                    or box[3] <= box[1]
                ):

                    continue

                detections.append((
                    box.astype(np.float64),
                    float(score)
                ))

        return detections

    def detect_rgb(self, rgb):

        raw = self.detect_raw(rgb)

        return non_max_suppression(
            raw,
            NMS_THRESHOLD
        )


# ==========================================
# KALMAN FILTER
# ==========================================

def box_to_measurement(box):

    x1, y1, x2, y2 = box

    return np.array([
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        max(1, x2 - x1),
        max(1, y2 - y1)
    ])


def measurement_to_box(state):

    cx, cy, w, h = state[:4]

    w = max(1, w)
    h = max(1, h)

    return np.array([
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2
    ])


class FaceTrack:

    def __init__(self, track_id, detection):

        box, score = detection

        self.id = track_id

        self.state = np.zeros(8)

        self.state[:4] = box_to_measurement(box)

        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)

        self.H = np.zeros((4, 8))
        self.H[:, :4] = np.eye(4)

        self.P = np.eye(8) * 100

        self.Q = np.diag([
            2, 2, 2, 2,
            4, 4, 2, 2
        ])

        self.R = np.diag([
            25, 25, 36, 36
        ])

        self.hits = 1
        self.missed = 0

        self.confidence = score

        self.best_frames = []

    @property
    def confirmed(self):

        return self.hits >= MIN_TRACK_HITS

    def box(self):

        return measurement_to_box(
            self.state
        )

    def predict(self):

        self.state = self.F @ self.state

        self.P = (
            self.F @ self.P @ self.F.T
            + self.Q
        )

        self.state[2:4] = np.maximum(
            self.state[2:4], 1
        )

        self.missed += 1

    def update(self, detection):

        box, score = detection

        z = box_to_measurement(box)

        innovation = (
            z - self.H @ self.state
        )

        S = (
            self.H @ self.P @ self.H.T
            + self.R
        )

        K = np.linalg.solve(
            S,
            self.H @ self.P
        ).T

        self.state += K @ innovation

        A = np.eye(8) - K @ self.H

        self.P = (
            A @ self.P @ A.T
            + K @ self.R @ K.T
        )

        self.state[2:4] = np.maximum(
            self.state[2:4], 1
        )

        self.hits += 1
        self.missed = 0
        self.confidence = score


# ==========================================
# MULTI-FACE TRACKING
# ==========================================

class MultiFaceTracker:

    def __init__(self):

        self.tracks = []
        self.all_tracks = []

        self.next_id = 1
        self.matches = 0

    def associate(
        self,
        track_indices,
        detections,
        limit
    ):

        if not track_indices or not detections:

            return (
                [],
                track_indices,
                list(range(len(detections)))
            )

        cost = np.full(
            (len(track_indices), len(detections)),
            1e6
        )

        for row, i in enumerate(track_indices):

            predicted = self.tracks[i].box()

            p = box_to_measurement(predicted)

            for col, (box, score) in enumerate(
                detections
            ):

                d = box_to_measurement(box)

                iou = calculate_iou(
                    predicted,
                    box
                )

                distance = (
                    np.linalg.norm(p[:2] - d[:2])
                    / max(np.linalg.norm(p[2:]), 1)
                )

                ratio = max(
                    p[2] / d[2],
                    d[2] / p[2],
                    p[3] / d[3],
                    d[3] / p[3]
                )

                if distance > 1.1 or ratio > 2.5:
                    continue

                if iou < 0.03 and distance > 0.4:
                    continue

                value = (
                    0.60 * (1 - iou)
                    + 0.30 * min(distance, 1)
                    + 0.10 * min(abs(np.log(ratio)), 1)
                )

                if value <= limit:

                    cost[row, col] = value

        rows, cols = linear_sum_assignment(cost)

        matches = []

        used_tracks = set()
        used_detections = set()

        for row, col in zip(rows, cols):

            if cost[row, col] > limit:
                continue

            track_index = track_indices[row]

            matches.append(
                (track_index, col)
            )

            used_tracks.add(track_index)
            used_detections.add(col)

        return (
            matches,

            [
                i for i in track_indices
                if i not in used_tracks
            ],

            [
                j for j in range(len(detections))
                if j not in used_detections
            ]
        )

    def update(self, detections):

        for track in self.tracks:

            track.predict()

        if detections is not None:

            high = [
                d for d in detections
                if d[1] >= HIGH_CONF
            ]

            low = [
                d for d in detections
                if LOW_CONF <= d[1] < HIGH_CONF
            ]

            indices = list(
                range(len(self.tracks))
            )

            matched, unmatched, unused = (
                self.associate(
                    indices,
                    high,
                    0.80
                )
            )

            for i, j in matched:

                self.tracks[i].update(high[j])

                self.matches += 1

            confirmed_unmatched = [
                i for i in unmatched
                if self.tracks[i].confirmed
            ]

            matched_low, _, _ = self.associate(
                confirmed_unmatched,
                low,
                0.68
            )

            for i, j in matched_low:

                self.tracks[i].update(low[j])

                self.matches += 1

            for j in unused:

                track = FaceTrack(
                    self.next_id,
                    high[j]
                )

                self.next_id += 1

                self.tracks.append(track)
                self.all_tracks.append(track)

        self.tracks = [
            track for track in self.tracks

            if track.missed <= (
                MAX_LOST_FRAMES
                if track.confirmed
                else 5
            )
        ]

        return [
            track for track in self.tracks

            if track.confirmed
            and track.missed <= MAX_VISIBLE_MISSED
        ]


# ==========================================
# BEST FACE FRAME SELECTION
# ==========================================

def save_best_candidate(
    track,
    rgb,
    frame_number
):

    h, w = rgb.shape[:2]

    x1, y1, x2, y2 = map(
        int, track.box()
    )

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(w, x2)
    y2 = min(h, y2)

    if (
        x2 - x1 < MIN_FACE_SIZE
        or y2 - y1 < MIN_FACE_SIZE
    ):

        return

    crop = rgb[y1:y2, x1:x2]

    gray = crop.astype(np.float32) @ np.array([
        0.299, 0.587, 0.114
    ])

    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4 * gray[1:-1, 1:-1]
    )

    blur = float(np.var(laplacian))
    brightness = float(gray.mean())

    if blur < MIN_BLUR_VARIANCE:
        return

    if not 40 <= brightness <= 220:
        return

    score = (
        0.40 * min(blur / 250, 1)
        + 0.35 * min(min(crop.shape[:2]) / 120, 1)
        + 0.25 * track.confidence
    )

    nearby = [
        item for item in track.best_frames

        if abs(
            item["frame"] - frame_number
        ) < MIN_BEST_FRAME_GAP
    ]

    if nearby:

        if max(
            item["score"] for item in nearby
        ) >= score:

            return

        remove = {
            item["frame"] for item in nearby
        }

        track.best_frames = [
            item for item in track.best_frames
            if item["frame"] not in remove
        ]

    track.best_frames.append({
        "frame": frame_number,
        "score": score,
        "crop": crop.copy()
    })

    track.best_frames.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    track.best_frames = (
        track.best_frames[:BEST_FRAMES_PER_TRACK]
    )


# ==========================================
# LIVE PREVIEW
# ==========================================

class LivePreview:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title(
            "SCRFD 10G - CPU Face Tracking"
        )

        self.root.configure(bg="#171717")

        self.label = tk.Label(
            self.root,
            bg="black"
        )

        self.label.pack(padx=8, pady=8)

        self.status = tk.Label(
            self.root,
            text="Initializing...",
            fg="white",
            bg="#171717"
        )

        self.status.pack(pady=8)

        self.frames = queue.Queue(maxsize=8)

        self.stop_event = threading.Event()

        self.displayed = 0
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

        self.root.after(5, self.refresh)

    def put(self, image, status):

        while not self.stop_event.is_set():

            try:

                self.frames.put(
                    (image, status),
                    timeout=0.1
                )

                return True

            except queue.Full:

                continue

        return False

    def process(self):

        try:

            OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            detector = SCRFDDetector()

            tracker = MultiFaceTracker()

            processed = 0
            decoded = 0
            detection_runs = 0

            detection_time = 0.0

            rows = []

            with av.open(str(VIDEO_PATH)) as source:

                fps = float(
                    source.streams.video[0].average_rate
                    or 24
                )

                start = time.perf_counter()

                for number, frame in enumerate(
                    source.decode(video=0)
                ):

                    if self.stop_event.is_set():
                        break

                    decoded += 1

                    target = start + number / fps

                    wait = target - time.perf_counter()

                    if wait > 0:

                        time.sleep(wait)

                    rgb = frame.to_ndarray(
                        format="rgb24"
                    )

                    detections = None

                    if number % DETECTION_INTERVAL == 0:

                        t0 = time.perf_counter()

                        detections = detector.detect_rgb(rgb)

                        detection_time += (
                            time.perf_counter() - t0
                        )

                        detection_runs += 1

                    tracks = tracker.update(
                        detections
                    )

                    image = Image.fromarray(rgb)

                    draw = ImageDraw.Draw(image)

                    for track in tracks:

                        if track.missed == 0:

                            save_best_candidate(
                                track,
                                rgb,
                                number
                            )

                        x1, y1, x2, y2 = map(
                            int,
                            track.box()
                        )

                        x1 = max(0, x1)
                        y1 = max(0, y1)

                        x2 = min(image.width - 1, x2)
                        y2 = min(image.height - 1, y2)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        draw.rectangle(
                            (x1, y1, x2, y2),
                            outline="#00ff66",
                            width=2
                        )

                        draw.text(
                            (x1, max(0, y1 - 15)),

                            f"ID {track.id} "
                            f"{track.confidence:.2f}",

                            fill="#00ff66",
                            stroke_width=1,
                            stroke_fill="black"
                        )

                        rows.append([
                            number,
                            track.id,
                            x1, y1, x2, y2,
                            track.confidence
                        ])

                    processed += 1

                    image.thumbnail(
                        (PREVIEW_MAX_WIDTH, 1000)
                    )

                    elapsed = (
                        time.perf_counter() - start
                    )

                    effective_fps = (
                        processed / max(elapsed, 1e-6)
                    )

                    status = (
                        f"FPS: {effective_fps:.2f} | "
                        f"Faces: {len(tracks)} | "
                        f"Frames: {processed} | "
                        f"Skipped: {decoded - processed}"
                    )

                    if not self.put(image, status):
                        break

            total_time = (
                time.perf_counter() - start
            )

            best_dir = OUTPUT_DIR / "best_faces"

            best_dir.mkdir(exist_ok=True)

            saved_faces = 0

            for track in tracker.all_tracks:

                if not track.confirmed:
                    continue

                for rank, item in enumerate(
                    track.best_frames, 1
                ):

                    filename = (
                        f"id_{track.id:03d}_"
                        f"rank_{rank}_"
                        f"frame_{item['frame']}.jpg"
                    )

                    Image.fromarray(
                        item["crop"]
                    ).save(
                        best_dir / filename,
                        quality=93
                    )

                    saved_faces += 1

            with open(
                OUTPUT_DIR / "tracks.csv",
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "frame",
                    "track_id",
                    "x1", "y1", "x2", "y2",
                    "confidence"
                ])

                writer.writerows(rows)

            report = {
                "model": "SCRFD 10G",

                "input_fps": fps,

                "decoded_frames": decoded,

                "processed_frames": processed,

                "skipped_frames": decoded - processed,

                "completed": not self.stop_event.is_set(),

                "detection_runs": detection_runs,

                "track_ids_created": tracker.next_id - 1,

                "successful_matches": tracker.matches,

                "best_faces_saved": saved_faces,

                "total_time_seconds": round(total_time, 2),

                "effective_fps": round(
                    processed / max(total_time, 1e-6),
                    2
                ),

                "average_detection_ms": round(
                    detection_time * 1000
                    / max(detection_runs, 1),
                    2
                ),

                "average_inference_ms": round(
                    detector.inference_time * 1000
                    / max(detector.inference_count, 1),
                    2
                )
            }

            with open(
                OUTPUT_DIR / "performance_report.json",
                "w"
            ) as file:

                json.dump(
                    report,
                    file,
                    indent=4
                )

            print("\n========== FINAL RESULTS ==========")

            for key, value in report.items():

                print(f"{key}: {value}")

            self.put(
                None,
                f"Completed | Frames: {processed}"
            )

        except Exception as error:

            print("ERROR:", repr(error))

            self.put(
                None,
                f"ERROR: {error}"
            )

    def refresh(self):

        try:

            image, status = self.frames.get_nowait()

            if image is not None:

                self.photo = ImageTk.PhotoImage(image)

                self.label.configure(
                    image=self.photo
                )

                self.displayed += 1

            self.status.configure(
                text=(
                    f"{status} | "
                    f"Displayed: {self.displayed}"
                )
            )

        except queue.Empty:

            pass

        if not self.stop_event.is_set():

            self.root.after(5, self.refresh)

    def close(self):

        self.stop_event.set()
        self.root.quit()

    def run(self):

        self.root.mainloop()

        self.stop_event.set()

        self.worker.join()

        self.root.destroy()


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(VIDEO_PATH)

    LivePreview().run()
