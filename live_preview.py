
import time
import queue
import threading
import tkinter as tk
from pathlib import Path
from fractions import Fraction

import av
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageTk, ImageFilter


# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "blaze_face_full_range_sparse.tflite"
VIDEO_PATH = BASE_DIR / "videos" / "input.mp4"
OUTPUT_PATH = BASE_DIR / "videos" / "output_live_preview.mp4"

SAVE_OUTPUT = True

# False: process every source frame, best for accuracy testing
# True: pace video to source FPS and skip late frames
REALTIME = False

# Detection
FULL_CONFIDENCE = 0.38
TILE_CONFIDENCE = 0.48
NEW_TRACK_CONFIDENCE = 0.52

TILE_SIZE = 400
TILE_OVERLAP = 0.25

# Number of additional BlazeFace calls per processed frame
TILES_PER_FRAME = 2

# Tracking
CONFIRM_HITS = 3
MAX_TENTATIVE_MISSED = 2
MAX_TRACK_MISSED = 12
MAX_VISIBLE_MISSED = 2

MATCH_IOU = 0.15
MATCH_DISTANCE = 0.55
SMOOTHING = 0.60

# Filtering
NMS_IOU = 0.35
MIN_FACE_SIZE = 10
MAX_FACE_RATIO = 0.65

# Display
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720
DISPLAY_QUEUE_SIZE = 2

# Recorder
RECORD_QUEUE_SIZE = 60


# =====================================================
# 2. CHECK FILES
# =====================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not VIDEO_PATH.is_file():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 3. GEOMETRY
# =====================================================

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

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def box_center(box):
    return np.array([
        (box[0] + box[2]) / 2,
        (box[1] + box[3]) / 2
    ], dtype=np.float32)


def box_diagonal(box):
    w = max(1, box[2] - box[0])
    h = max(1, box[3] - box[1])
    return float(np.hypot(w, h))


def valid_box(box, frame_w, frame_h):
    w = box[2] - box[0]
    h = box[3] - box[1]

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False

    if w > frame_w * MAX_FACE_RATIO:
        return False

    if h > frame_h * MAX_FACE_RATIO:
        return False

    aspect = w / max(h, 1)

    return 0.45 <= aspect <= 1.8


# =====================================================
# 4. DUPLICATE REMOVAL
# =====================================================

def non_max_suppression(detections):
    ordered = sorted(
        detections,
        key=lambda d: d["confidence"],
        reverse=True
    )

    kept = []

    for detection in ordered:
        duplicate = False

        for existing in kept:
            if calculate_iou(
                detection["box"],
                existing["box"]
            ) >= NMS_IOU:
                duplicate = True
                break

        if not duplicate:
            kept.append(detection)

    return kept


# =====================================================
# 5. BLAZEFACE DETECTOR
# =====================================================

def create_detector(confidence):
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=mp.tasks.BaseOptions.Delegate.CPU
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=confidence
    )

    return mp.tasks.vision.FaceDetector.create_from_options(
        options
    )


def run_detector(
    detector,
    rgb,
    offset_x=0,
    offset_y=0
):
    h, w = rgb.shape[:2]

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb)
    )

    result = detector.detect(image)

    detections = []

    for item in result.detections:
        if not item.categories:
            continue

        bbox = item.bounding_box

        x1 = max(0, float(bbox.origin_x))
        y1 = max(0, float(bbox.origin_y))

        x2 = min(
            w,
            float(bbox.origin_x + bbox.width)
        )

        y2 = min(
            h,
            float(bbox.origin_y + bbox.height)
        )

        local_box = np.array(
            [x1, y1, x2, y2],
            dtype=np.float32
        )

        if not valid_box(local_box, w, h):
            continue

        global_box = local_box + np.array(
            [
                offset_x,
                offset_y,
                offset_x,
                offset_y
            ],
            dtype=np.float32
        )

        detections.append({
            "box": global_box,
            "confidence": float(
                item.categories[0].score
            )
        })

    return detections


# =====================================================
# 6. TILE SCHEDULER
# =====================================================

def tile_positions(length, size):
    if length <= size:
        return [0]

    step = max(
        1,
        int(size * (1 - TILE_OVERLAP))
    )

    positions = list(
        range(0, length - size + 1, step)
    )

    last = length - size

    if positions[-1] != last:
        positions.append(last)

    return positions


class TileScheduler:

    def __init__(self, width, height):
        self.width = width
        self.height = height

        tw = min(TILE_SIZE, width)
        th = min(TILE_SIZE, height)

        self.regions = []

        for y in tile_positions(height, th):
            for x in tile_positions(width, tw):
                if tw == width and th == height:
                    continue

                self.regions.append(
                    (x, y, tw, th)
                )

        self.cursor = 0

    def next_regions(self, count):
        if not self.regions:
            return []

        selected = []

        for _ in range(min(count, len(self.regions))):
            region = self.regions[
                self.cursor % len(self.regions)
            ]

            self.cursor += 1
            selected.append(region)

        return selected


# =====================================================
# 7. SIMPLE KALMAN BOX FILTER
# =====================================================

class KalmanBox:

    def __init__(self, box):
        cx, cy = box_center(box)

        w = box[2] - box[0]
        h = box[3] - box[1]

        self.state = np.array(
            [cx, cy, w, h, 0, 0, 0, 0],
            dtype=np.float64
        )

        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)

        self.H = np.eye(4, 8)

        self.P = np.eye(8) * 100.0
        self.Q = np.eye(8) * 1.0
        self.R = np.eye(4) * 10.0

    def predict(self):
        self.state = self.F @ self.state

        self.P = (
            self.F @ self.P @ self.F.T
            + self.Q
        )

        self.state[2:4] = np.maximum(
            self.state[2:4],
            4
        )

    def update(self, box):
        cx, cy = box_center(box)

        z = np.array([
            cx,
            cy,
            box[2] - box[0],
            box[3] - box[1]
        ], dtype=np.float64)

        innovation = z - self.H @ self.state

        S = (
            self.H @ self.P @ self.H.T
            + self.R
        )

        K = (
            self.P @ self.H.T
            @ np.linalg.inv(S)
        )

        self.state += K @ innovation

        identity = np.eye(8)

        self.P = (
            (identity - K @ self.H)
            @ self.P
            @ (identity - K @ self.H).T
            + K @ self.R @ K.T
        )

        self.state[2:4] = np.maximum(
            self.state[2:4],
            4
        )

    @property
    def box(self):
        cx, cy, w, h = self.state[:4]

        return np.array([
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2
        ], dtype=np.float32)


# =====================================================
# 8. TEMPORAL FACE TRACKER
# =====================================================

class FaceTracker:

    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        for track in self.tracks:
            track["filter"].predict()
            track["updated"] = False

        candidates = []

        for ti, track in enumerate(self.tracks):
            predicted = track["filter"].box

            for di, detection in enumerate(detections):
                box = detection["box"]

                overlap = calculate_iou(
                    predicted,
                    box
                )

                distance = np.linalg.norm(
                    box_center(predicted)
                    - box_center(box)
                )

                normalized_distance = distance / max(
                    box_diagonal(predicted),
                    box_diagonal(box)
                )

                size_ratio = (
                    box_diagonal(box)
                    / max(1, box_diagonal(predicted))
                )

                if not 0.5 <= size_ratio <= 2.0:
                    continue

                if (
                    overlap >= MATCH_IOU
                    or normalized_distance <= MATCH_DISTANCE
                ):
                    cost = (
                        0.6 * (1 - overlap)
                        + 0.4 * normalized_distance
                    )

                    candidates.append(
                        (cost, ti, di)
                    )

        candidates.sort()

        used_tracks = set()
        used_detections = set()

        for _, ti, di in candidates:
            if ti in used_tracks:
                continue

            if di in used_detections:
                continue

            track = self.tracks[ti]
            detection = detections[di]

            track["filter"].update(
                detection["box"]
            )

            track["confidence"] = (
                0.7 * track["confidence"]
                + 0.3 * detection["confidence"]
            )

            track["hits"] += 1
            track["missed"] = 0
            track["updated"] = True

            if (
                not track["confirmed"]
                and track["hits"] >= CONFIRM_HITS
            ):
                track["confirmed"] = True
                track["id"] = self.next_id
                self.next_id += 1

            used_tracks.add(ti)
            used_detections.add(di)

        for track in self.tracks:
            if not track["updated"]:
                track["missed"] += 1

        # New tracks must start with stronger evidence.
        for di, detection in enumerate(detections):
            if di in used_detections:
                continue

            if (
                detection["confidence"]
                < NEW_TRACK_CONFIDENCE
            ):
                continue

            self.tracks.append({
                "filter": KalmanBox(detection["box"]),
                "confidence": detection["confidence"],
                "hits": 1,
                "missed": 0,
                "confirmed": False,
                "updated": True,
                "id": None
            })

        self.tracks = [
            track
            for track in self.tracks
            if track["missed"] <= (
                MAX_TRACK_MISSED
                if track["confirmed"]
                else MAX_TENTATIVE_MISSED
            )
        ]

        visible = []

        for track in self.tracks:
            if not track["confirmed"]:
                continue

            if track["missed"] > MAX_VISIBLE_MISSED:
                continue

            visible.append({
                "id": track["id"],
                "box": track["filter"].box,
                "confidence": track["confidence"],
                "missed": track["missed"]
            })

        return visible


# =====================================================
# 9. ANNOTATION - PILLOW, NO OPENCV
# =====================================================

def annotate_frame(
    rgb,
    tracks,
    fps,
    detection_ms,
    dropped
):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    width, height = image.size

    active_faces = 0

    for track in tracks:
        box = track["box"]

        x1 = int(max(0, min(width - 1, box[0])))
        y1 = int(max(0, min(height - 1, box[1])))
        x2 = int(max(0, min(width - 1, box[2])))
        y2 = int(max(0, min(height - 1, box[3])))

        if x2 <= x1 or y2 <= y1:
            continue

        if track["missed"] == 0:
            color = "lime"
            active_faces += 1
        else:
            color = "orange"

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=3
        )

        label = (
            f"ID {track['id']} "
            f"{track['confidence']:.0%}"
        )

        if track["missed"]:
            label += " HOLD"

        draw.text(
            (x1, max(0, y1 - 16)),
            label,
            fill=color,
            stroke_width=1,
            stroke_fill="black"
        )

    draw.rectangle(
        (0, 0, min(width, 760), 38),
        fill="black"
    )

    info = (
        f"Faces: {active_faces} | "
        f"FPS: {fps:.1f} | "
        f"Detect: {detection_ms:.1f} ms | "
        f"Dropped: {dropped}"
    )

    draw.text(
        (10, 12),
        info,
        fill="yellow"
    )

    return image


# =====================================================
# 10. MP4 VIDEO RECORDER
# =====================================================

class VideoRecorder:

    def __init__(self, path, fps):
        self.path = path
        self.fps = Fraction(fps).limit_denominator(100000)

        self.frames = queue.Queue(
            maxsize=RECORD_QUEUE_SIZE
        )

        self.error = None
        self.saved_frames = 0
        self.closed = False

        self.thread = threading.Thread(
            target=self._write,
            daemon=False
        )

        self.thread.start()

    def add_frame(self, index, image):
        while True:
            if self.error:
                raise RuntimeError(self.error)

            if not self.thread.is_alive():
                raise RuntimeError(
                    "Recorder thread stopped"
                )

            try:
                self.frames.put(
                    ("frame", index, image),
                    timeout=0.1
                )
                return

            except queue.Full:
                pass

    def close(self, total_frames):
        if self.closed:
            return

        self.closed = True

        if self.thread.is_alive():
            while self.thread.is_alive():
                try:
                    self.frames.put(
                        ("end", total_frames, None),
                        timeout=0.1
                    )
                    break

                except queue.Full:
                    pass

        self.thread.join()

        if self.error:
            raise RuntimeError(self.error)

    def _write(self):
        container = None

        try:
            container = av.open(
                str(self.path),
                mode="w"
            )

            stream = None
            previous_rgb = None
            next_index = 0

            def encode_rgb(rgb):
                frame = av.VideoFrame.from_ndarray(
                    rgb,
                    format="rgb24"
                )

                for packet in stream.encode(frame):
                    container.mux(packet)

                self.saved_frames += 1

            while True:
                kind, index, image = self.frames.get()

                if kind == "end":
                    if previous_rgb is not None:
                        while next_index < index:
                            encode_rgb(previous_rgb)
                            next_index += 1
                    break

                rgb = np.asarray(
                    image.convert("RGB")
                )

                height, width = rgb.shape[:2]

                rgb = np.ascontiguousarray(
                    rgb[
                        :height - height % 2,
                        :width - width % 2
                    ]
                )

                if stream is None:
                    try:
                        stream = container.add_stream(
                            "libx264",
                            rate=self.fps
                        )

                        stream.options = {
                            "crf": "18",
                            "preset": "veryfast"
                        }

                    except Exception:
                        stream = container.add_stream(
                            "mpeg4",
                            rate=self.fps
                        )
                        stream.bit_rate = 4000000

                    stream.width = rgb.shape[1]
                    stream.height = rgb.shape[0]
                    stream.pix_fmt = "yuv420p"

                    previous_rgb = rgb

                if rgb.shape != previous_rgb.shape:
                    raise RuntimeError(
                        "Video resolution changed"
                    )

                # Preserve timeline when frames are skipped.
                while next_index < index:
                    encode_rgb(previous_rgb)
                    next_index += 1

                encode_rgb(rgb)

                previous_rgb = rgb
                next_index = index + 1

            if stream is not None:
                for packet in stream.encode():
                    container.mux(packet)

        except Exception as error:
            self.error = str(error)

        finally:
            if container is not None:
                container.close()


# =====================================================
# 11. LIVE PREVIEW APPLICATION
# =====================================================

class LivePreviewApp:

    def __init__(self, root):
        self.root = root

        self.root.title(
            "BlazeFace CCTV - Face Detection"
        )

        self.root.configure(bg="#151515")

        self.video_label = tk.Label(
            root,
            bg="black"
        )

        self.video_label.pack(
            padx=10,
            pady=10
        )

        self.status = tk.Label(
            root,
            text="Starting detection...",
            fg="white",
            bg="#151515",
            font=("Arial", 11)
        )

        self.status.pack(pady=5)

        self.stop_event = threading.Event()

        self.frame_queue = queue.Queue(
            maxsize=DISPLAY_QUEUE_SIZE
        )

        self.worker = threading.Thread(
            target=self.process_video,
            daemon=False
        )

        self.worker.start()

        self.root.after(
            15,
            self.update_display
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

    def send_message(self, message):
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()

            self.frame_queue.put_nowait(message)

        except (queue.Empty, queue.Full):
            pass

    # =================================================
    # MAIN PROCESSING
    # =================================================

    def process_video(self):
        container = None
        full_detector = None
        tile_detector = None
        recorder = None

        tracker = FaceTracker()
        scheduler = None

        processed = 0
        dropped = 0
        frame_index = 0

        detection_times = []
        processing_times = []

        try:
            container = av.open(
                str(VIDEO_PATH)
            )

            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            rate = (
                stream.average_rate
                or Fraction(30, 1)
            )

            fps = float(rate)

            if fps <= 0:
                rate = Fraction(30, 1)
                fps = 30.0

            interval = 1.0 / fps

            full_detector = create_detector(
                FULL_CONFIDENCE
            )

            tile_detector = create_detector(
                TILE_CONFIDENCE
            )

            if SAVE_OUTPUT:
                recorder = VideoRecorder(
                    OUTPUT_PATH,
                    rate
                )

            playback_start = time.perf_counter()
            run_start = playback_start

            previous_time = None
            processing_fps = 0.0

            for frame in container.decode(video=0):
                if self.stop_event.is_set():
                    break

                target_time = (
                    playback_start
                    + frame_index * interval
                )

                if REALTIME:
                    now = time.perf_counter()

                    if now - target_time > interval:
                        dropped += 1
                        frame_index += 1
                        continue

                    remaining = target_time - now

                    if remaining > 0:
                        if self.stop_event.wait(remaining):
                            break

                start = time.perf_counter()

                rgb = np.ascontiguousarray(
                    frame.to_ndarray(
                        format="rgb24"
                    )
                )

                height, width = rgb.shape[:2]

                if scheduler is None:
                    scheduler = TileScheduler(
                        width,
                        height
                    )

                # Full-frame detection
                detect_start = time.perf_counter()

                detections = run_detector(
                    full_detector,
                    rgb
                )

                # Additional tiled detection
                for x, y, tw, th in scheduler.next_regions(
                    TILES_PER_FRAME
                ):
                    crop = rgb[
                        y:y + th,
                        x:x + tw
                    ]

                    tile_detections = run_detector(
                        tile_detector,
                        crop,
                        x,
                        y
                    )

                    for detection in tile_detections:
                        box = detection["box"]

                        face_w = box[2] - box[0]
                        face_h = box[3] - box[1]

                        if (
                            face_w > tw * 0.40
                            or face_h > th * 0.40
                        ):
                            continue

                        # Reject faces cut by an internal tile edge.
                        margin = 3

                        if (
                            (x > 0 and box[0] <= x + margin)
                            or (y > 0 and box[1] <= y + margin)
                            or (
                                x + tw < width
                                and box[2] >= x + tw - margin
                            )
                            or (
                                y + th < height
                                and box[3] >= y + th - margin
                            )
                        ):
                            continue

                        detections.append(detection)

                detections = non_max_suppression(
                    detections
                )

                detection_ms = (
                    time.perf_counter()
                    - detect_start
                ) * 1000

                detection_times.append(
                    detection_ms
                )

                tracks = tracker.update(
                    detections
                )

                current_time = time.perf_counter()

                if previous_time is not None:
                    elapsed = current_time - previous_time

                    if elapsed > 0:
                        instant_fps = 1.0 / elapsed

                        processing_fps = (
                            instant_fps
                            if processing_fps == 0
                            else (
                                0.2 * instant_fps
                                + 0.8 * processing_fps
                            )
                        )

                previous_time = current_time

                annotated = annotate_frame(
                    rgb,
                    tracks,
                    processing_fps,
                    detection_ms,
                    dropped
                )

                if recorder is not None:
                    recorder.add_frame(
                        frame_index,
                        annotated
                    )

                preview = annotated.copy()

                preview.thumbnail(
                    (
                        PREVIEW_WIDTH,
                        PREVIEW_HEIGHT
                    ),
                    Image.Resampling.BILINEAR
                )

                self.send_message(
                    (
                        "frame",
                        preview,
                        frame_index + 1
                    )
                )

                processing_ms = (
                    time.perf_counter() - start
                ) * 1000

                processing_times.append(
                    processing_ms
                )

                processed += 1
                frame_index += 1

            if recorder is not None:
                recorder.close(frame_index)
                recorder = None

            wall_time = (
                time.perf_counter() - run_start
            )

            print("\n" + "=" * 50)
            print("BLAZEFACE PERFORMANCE REPORT")
            print("=" * 50)

            print(f"Input FPS: {fps:.2f}")
            print(f"Processed frames: {processed}")
            print(f"Dropped frames: {dropped}")

            if processed and wall_time > 0:
                print(
                    f"Average throughput: "
                    f"{processed / wall_time:.2f} FPS"
                )

            if detection_times:
                print(
                    f"Detection average: "
                    f"{np.mean(detection_times):.2f} ms"
                )

                print(
                    f"Detection P95: "
                    f"{np.percentile(detection_times, 95):.2f} ms"
                )

            if processing_times:
                print(
                    f"Processing average: "
                    f"{np.mean(processing_times):.2f} ms"
                )

                print(
                    f"Processing P95: "
                    f"{np.percentile(processing_times, 95):.2f} ms"
                )

            print(
                f"Confirmed IDs created: "
                f"{tracker.next_id - 1}"
            )

            if SAVE_OUTPUT:
                print(
                    f"Output saved: {OUTPUT_PATH}"
                )

            print("=" * 50)

            self.send_message(
                ("done", processed, dropped)
            )

        except Exception as error:
            print("ERROR:", error)

            self.send_message(
                ("error", str(error))
            )

        finally:
            if recorder is not None:
                try:
                    recorder.close(frame_index)
                except Exception as error:
                    print(
                        "Recorder error:",
                        error
                    )

            if full_detector is not None:
                full_detector.close()

            if tile_detector is not None:
                tile_detector.close()

            if container is not None:
                container.close()

    # =================================================
    # DISPLAY
    # =================================================

    def update_display(self):
        latest = None
        terminal_message = None

        try:
            while True:
                item = self.frame_queue.get_nowait()

                if item[0] == "frame":
                    latest = item

                elif item[0] == "done":
                    terminal_message = (
                        f"Finished | "
                        f"Processed: {item[1]} | "
                        f"Dropped: {item[2]}"
                    )

                elif item[0] == "error":
                    terminal_message = (
                        f"Error: {item[1]}"
                    )

        except queue.Empty:
            pass

        if latest is not None:
            photo = ImageTk.PhotoImage(
                latest[1]
            )

            self.video_label.configure(
                image=photo
            )

            self.video_label.image = photo

            self.status.configure(
                text=(
                    f"Frame: {latest[2]} | "
                    f"Realtime: {REALTIME} | "
                    f"Recording: {SAVE_OUTPUT}"
                )
            )

        if terminal_message is not None:
            self.status.configure(
                text=terminal_message
            )

        if (
            not self.stop_event.is_set()
            or self.worker.is_alive()
        ):
            self.root.after(
                15,
                self.update_display
            )

    # =================================================
    # CLOSE
    # =================================================

    def close(self):
        self.stop_event.set()

        if self.worker.is_alive():
            self.status.configure(
                text="Finalizing MP4... Please wait."
            )

            self.root.after(
                100,
                self.close
            )
        else:
            self.root.destroy()


# =====================================================
# 12. START
# =====================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = LivePreviewApp(root)
    root.mainloop()
