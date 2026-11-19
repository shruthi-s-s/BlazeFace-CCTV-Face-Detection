
import time
import threading
import queue
import tkinter as tk
import traceback

import av
import numpy as np
import comtypes

from PIL import Image, ImageDraw, ImageTk
from pygrabber.dshow_graph import FilterGraph
from scrfd_live_preview import SCRFDDetector


# ==========================================
# CONFIGURATION
# ==========================================

CAMERA_INDEX = 0

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

MIRROR_VIEW = True

CONFIDENCE_THRESHOLD = 0.50

# ==========================================
# NORMALIZE DETECTOR OUTPUT
# ==========================================

def normalize_detections(result):

    detections = []

    if result is None:
        return detections

    # Support tuple: boxes, scores

    if isinstance(result, tuple) and len(result) >= 2:

        boxes = np.asarray(result[0])
        scores = np.asarray(result[1])

        if boxes.ndim == 2 and boxes.shape[1] == 4:

            if scores.size == len(boxes):

                scores = scores.reshape(-1)

                return [
                    (box, float(score))
                    for box, score in zip(boxes, scores)
                ]

        # Support boxes containing confidence

        if boxes.ndim == 2 and boxes.shape[1] >= 5:

            return [
                (row[:4], float(row[4]))
                for row in boxes
            ]

    if isinstance(result, dict):
        result = [result]

    # Support list of detections or Nx5 arrays

    for item in result:

        if isinstance(item, dict):

            box = item.get("bbox")

            if box is None:
                box = item.get("box")

            score = item.get("score")

            if score is None:
                score = item.get("confidence")

            if box is not None and score is not None:

                detections.append(
                    (box, float(score))
                )

            continue

        if isinstance(item, (tuple, list)):

            if len(item) >= 2:

                box = np.asarray(item[0])

                if box.size == 4:

                    detections.append(
                        (
                            box.reshape(4),
                            float(item[1])
                        )
                    )

                    continue

        row = np.asarray(item).reshape(-1)

        if row.size >= 5:

            detections.append(
                (row[:4], float(row[4]))
            )

        else:
            raise ValueError(
                f"Unexpected detection format: {item}"
            )

    return detections


# ==========================================
# WEBCAM APPLICATION
# ==========================================

class WebcamApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "SCRFD 10G - Live Webcam Face Detection"
        )

        self.root.configure(bg="#151515")

        self.stop_event = threading.Event()

        self.frame_queue = queue.Queue(maxsize=1)

        self.video_label = tk.Label(
            root,
            bg="black",
            fg="white",
            text="Starting webcam...",
            width=80,
            height=20
        )

        self.video_label.pack(
            padx=10,
            pady=10
        )

        self.status = tk.Label(
            root,
            text="Loading SCRFD 10G...",
            font=("Arial", 12),
            fg="white",
            bg="#151515"
        )

        self.status.pack(pady=5)

        tk.Button(
            root,
            text="Stop Webcam",
            command=self.close,
            font=("Arial", 11),
            bg="#cc3333",
            fg="white"
        ).pack(pady=10)

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

        self.root.bind(
            "<Escape>",
            lambda event: self.close()
        )

        self.worker = threading.Thread(
            target=self.process_webcam,
            daemon=True
        )

        self.worker.start()

        self.update_display()

    # ======================================
    # CAMERA INITIALIZATION
    # ======================================

    def get_camera(self):

        cameras = FilterGraph().get_input_devices()

        if not cameras:
            raise RuntimeError(
                "No webcam detected."
            )

        print("\nAvailable cameras:")

        for index, camera in enumerate(cameras):

            print(index, camera)

        if not 0 <= CAMERA_INDEX < len(cameras):

            raise RuntimeError(
                "Invalid camera index."
            )

        camera_name = cameras[CAMERA_INDEX]

        print("\nUsing camera:", camera_name)

        return camera_name

    # ======================================
    # SEND FRAME TO GUI
    # ======================================

    def publish(self, image, status):

        try:
            self.frame_queue.get_nowait()

        except queue.Empty:
            pass

        try:
            self.frame_queue.put_nowait(
                (image, status)
            )

        except queue.Full:
            pass

    # ======================================
    # PROCESS WEBCAM
    # ======================================

    def process_webcam(self):

        container = None

        com_initialized = False

        processed_frames = 0

        total_detection_ms = 0.0

        start_time = None

        try:

            # Fix Windows COM initialization

            comtypes.CoInitialize()

            com_initialized = True

            # Load existing SCRFD 10G detector

            detector = SCRFDDetector()

            camera_name = self.get_camera()

            container = av.open(
                f"video={camera_name}",
                format="dshow",
                options={
                    "rtbufsize": "32M"
                }
            )

            start_time = time.perf_counter()

            print("\n==========================")
            print("LIVE WEBCAM STARTED")
            print("Model: SCRFD 10G")
            print("CPU-only processing")
            print("No OpenCV")
            print("No face recognition")
            print("==========================\n")

            for frame in container.decode(video=0):

                if self.stop_event.is_set():
                    break

                # Convert webcam frame into RGB

                rgb = frame.to_ndarray(
                    format="rgb24"
                )

                if MIRROR_VIEW:

                    rgb = np.ascontiguousarray(
                        rgb[:, ::-1]
                    )

                # ==============================
                # CORRECT DETECTION METHOD
                # ==============================

                detection_start = time.perf_counter()

                result = detector.detect_rgb(rgb)

                detection_ms = (
                    time.perf_counter()
                    - detection_start
                ) * 1000

                total_detection_ms += detection_ms

                detections = normalize_detections(
                    result
                )

                # Convert RGB array into PIL image

                image = Image.fromarray(rgb)

                draw = ImageDraw.Draw(image)

                width, height = image.size

                face_count = 0

                # ==============================
                # DRAW BOUNDING BOXES
                # ==============================

                for box, score in detections:

                    if score < CONFIDENCE_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(
                        int, box
                    )

                    x1 = max(
                        0, min(width - 1, x1)
                    )

                    y1 = max(
                        0, min(height - 1, y1)
                    )

                    x2 = max(
                        0, min(width - 1, x2)
                    )

                    y2 = max(
                        0, min(height - 1, y2)
                    )

                    if x2 <= x1 or y2 <= y1:
                        continue

                    face_count += 1

                    # Green face rectangle

                    draw.rectangle(
                        (x1, y1, x2, y2),
                        outline="#00FF66",
                        width=3
                    )

                    label = (
                        f"Face {score * 100:.1f}%"
                    )

                    text_y = max(0, y1 - 20)

                    draw.rectangle(
                        (
                            x1,
                            text_y,
                            min(width - 1, x1 + 110),
                            text_y + 19
                        ),
                        fill="#006633"
                    )

                    draw.text(
                        (x1 + 3, text_y + 2),
                        label,
                        fill="white"
                    )

                # ==============================
                # PERFORMANCE METRICS
                # ==============================

                processed_frames += 1

                elapsed = max(
                    time.perf_counter() - start_time,
                    1e-6
                )

                fps = processed_frames / elapsed

                average_ms = (
                    total_detection_ms /
                    processed_frames
                )

                # Draw performance overlay

                draw.rectangle(
                    (0, 0, min(width, 440), 85),
                    fill="#151515"
                )

                draw.text(
                    (10, 8),
                    "SCRFD 10G | CPU ONLY",
                    fill="#00FF66"
                )

                draw.text(
                    (10, 28),
                    f"Faces: {face_count}",
                    fill="white"
                )

                draw.text(
                    (10, 45),
                    f"FPS: {fps:.2f}",
                    fill="white"
                )

                draw.text(
                    (10, 62),
                    f"Detection: {detection_ms:.2f} ms",
                    fill="white"
                )

                status = (
                    f"Faces: {face_count}   |   "
                    f"FPS: {fps:.2f}   |   "
                    f"Latency: {detection_ms:.2f} ms   |   "
                    f"Average: {average_ms:.2f} ms"
                )

                image.thumbnail(
                    (DISPLAY_WIDTH, DISPLAY_HEIGHT)
                )

                self.publish(image, status)

                if processed_frames % 30 == 0:

                    print(
                        f"Frames: {processed_frames} | "
                        f"Faces: {face_count} | "
                        f"FPS: {fps:.2f} | "
                        f"Latency: {detection_ms:.2f} ms"
                    )

        except Exception as error:

            traceback.print_exc()

            self.publish(
                None,
                f"Error: {error}"
            )

        finally:

            try:

                if container is not None:
                    container.close()

            finally:

                if com_initialized:
                    comtypes.CoUninitialize()

            print("\n==========================")
            print("WEBCAM SUMMARY")
            print("==========================")

            print(
                "Processed frames:",
                processed_frames
            )

            if processed_frames and start_time:

                elapsed = max(
                    time.perf_counter() - start_time,
                    1e-6
                )

                print(
                    "Average detection latency:",
                    round(
                        total_detection_ms /
                        processed_frames,
                        2
                    ),
                    "ms"
                )

                print(
                    "Effective FPS:",
                    round(
                        processed_frames / elapsed,
                        2
                    )
                )

    # ======================================
    # LIVE DISPLAY
    # ======================================

    def update_display(self):

        if self.stop_event.is_set():
            return

        try:

            image, status = (
                self.frame_queue.get_nowait()
            )

            if image is not None:

                photo = ImageTk.PhotoImage(image)

                self.video_label.configure(
                    image=photo,
                    text="",
                    width=image.width,
                    height=image.height
                )

                self.video_label.image = photo

            self.status.configure(
                text=status
            )

        except queue.Empty:
            pass

        self.root.after(
            10,
            self.update_display
        )

    # ======================================
    # STOP CAMERA
    # ======================================

    def close(self):

        self.stop_event.set()

        self.root.destroy()


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    root = tk.Tk()

    app = WebcamApp(root)

    root.mainloop()

    app.worker.join(timeout=3)
