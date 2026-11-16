
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "2.5g_bnkps.onnx"
IMAGE_PATH = BASE_DIR / "test.jpg"
OUTPUT_PATH = BASE_DIR / "scrfd_result.jpg"

INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.50
NMS_THRESHOLD = 0.40


def preprocess(image):
    original_w, original_h = image.size

    scale = min(
        INPUT_SIZE / original_w,
        INPUT_SIZE / original_h
    )

    new_w = max(1, round(original_w * scale))
    new_h = max(1, round(original_h * scale))

    resized = image.convert("RGB").resize(
        (new_w, new_h),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "RGB", (INPUT_SIZE, INPUT_SIZE), (0, 0, 0)
    )
    canvas.paste(resized, (0, 0))

    array = np.asarray(canvas, dtype=np.float32)

    # SCRFD expects BGR channel order.
    array = array[:, :, ::-1]

    array = (array - 127.5) / 128.0
    array = array.transpose(2, 0, 1)
    array = np.ascontiguousarray(
        array[np.newaxis, ...], dtype=np.float32
    )

    return array, scale


def decode_outputs(outputs, scale):
    all_boxes = []
    all_scores = []

    for index, stride in enumerate((8, 16, 32)):
        scores = outputs[index].reshape(-1)
        distances = outputs[index + 3].reshape(-1, 4)

        grid_h = INPUT_SIZE // stride
        grid_w = INPUT_SIZE // stride

        yy, xx = np.mgrid[0:grid_h, 0:grid_w]

        centers = np.stack(
            (xx, yy), axis=-1
        ).reshape(-1, 2)

        # Two anchors per grid location.
        centers = np.repeat(centers, 2, axis=0)
        centers = centers.astype(np.float32) * stride

        if len(scores) != len(centers):
            raise ValueError(
                f"Unexpected anchor count at stride {stride}"
            )

        selected = scores >= CONFIDENCE_THRESHOLD

        if not np.any(selected):
            continue

        selected_centers = centers[selected]
        selected_distances = distances[selected] * stride

        x1y1 = (
            selected_centers -
            selected_distances[:, :2]
        )
        x2y2 = (
            selected_centers +
            selected_distances[:, 2:]
        )

        boxes = np.concatenate(
            (x1y1, x2y2), axis=1
        ) / scale

        all_boxes.append(boxes)
        all_scores.append(scores[selected])

    if not all_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32)
        )

    return (
        np.concatenate(all_boxes),
        np.concatenate(all_scores)
    )


def non_max_suppression(boxes, scores):
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T

    areas = (
        np.maximum(0, x2 - x1) *
        np.maximum(0, y2 - y1)
    )

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        current = order[0]
        keep.append(current)

        if len(order) == 1:
            break

        remaining = order[1:]

        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])

        intersection = (
            np.maximum(0, xx2 - xx1) *
            np.maximum(0, yy2 - yy1)
        )

        union = (
            areas[current] +
            areas[remaining] -
            intersection
        )

        iou = intersection / np.maximum(union, 1e-6)

        order = remaining[iou <= NMS_THRESHOLD]

    return keep


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Place a test image here: {IMAGE_PATH}"
        )

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    image = Image.open(IMAGE_PATH).convert("RGB")
    input_tensor, scale = preprocess(image)

    input_name = session.get_inputs()[0].name

    start = time.perf_counter()

    outputs = session.run(
        None,
        {input_name: input_tensor}
    )

    inference_ms = (
        time.perf_counter() - start
    ) * 1000

    boxes, scores = decode_outputs(outputs, scale)
    selected_indices = non_max_suppression(boxes, scores)

    draw = ImageDraw.Draw(image)

    for index in selected_indices:
        box = boxes[index]
        score = scores[index]

        x1, y1, x2, y2 = box

        x1 = max(0, min(image.width, float(x1)))
        y1 = max(0, min(image.height, float(y1)))
        x2 = max(0, min(image.width, float(x2)))
        y2 = max(0, min(image.height, float(y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="lime",
            width=3
        )

        draw.text(
            (x1, max(0, y1 - 15)),
            f"Face {score:.2f}",
            fill="lime"
        )

    image.save(OUTPUT_PATH)

    print("SCRFD detection completed!")
    print("Faces detected:", len(selected_indices))
    print(f"Model inference time: {inference_ms:.2f} ms")
    print("Output saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
