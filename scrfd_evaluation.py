
import json
import csv
import time

from pathlib import Path
from collections import defaultdict

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from scrfd_live_preview import (
    SCRFDDetector,
    calculate_iou,
    non_max_suppression
)


# ==========================================
# CONFIGURATION
# ==========================================

BASE = Path(__file__).resolve().parent

IMAGE_DIR = BASE / "evaluation" / "frames"

ANNOTATION_PATH = (
    BASE / "evaluation" / "instances_default.json"
)

OUTPUT_DIR = (
    BASE / "evaluation" / "results" / "SCRFD10G_optimized"
)

VISUAL_DIR = OUTPUT_DIR / "comparison_images"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VISUAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IOU_THRESHOLD = 0.50

MIN_CONFIDENCE = 0.05

CONF_THRESHOLDS = [
    round(float(value), 2)
    for value in np.arange(0.05, 0.96, 0.05)
]

NMS_THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55
]

EXPERIMENTS = [
    {
        "name": "RGB",
        "rgb": True,
        "clahe": False
    },
    {
        "name": "BGR",
        "rgb": False,
        "clahe": False
    },
    {
        "name": "RGB_CLAHE",
        "rgb": True,
        "clahe": True
    },
    {
        "name": "BGR_CLAHE",
        "rgb": False,
        "clahe": True
    }
]


# ==========================================
# LOAD ANNOTATIONS
# ==========================================

if not ANNOTATION_PATH.exists():

    raise FileNotFoundError(
        ANNOTATION_PATH
    )

coco = COCO(
    str(ANNOTATION_PATH)
)

coco.dataset.setdefault(
    "info", {}
)

images = coco.dataset["images"]

if not images:

    raise ValueError(
        "No evaluation images found."
    )

category_ids = [
    category["id"]

    for category in coco.dataset["categories"]

    if category["name"].strip().lower() == "face"
]

if len(category_ids) != 1:

    raise ValueError(
        "Expected one Face category."
    )

category_id = category_ids[0]

image_lookup = {
    image["id"]: image
    for image in images
}

ground_truth = defaultdict(list)

for annotation in coco.dataset["annotations"]:

    if annotation["category_id"] != category_id:
        continue

    if annotation.get("iscrowd", 0):

        raise ValueError(
            "Crowd annotations are not supported."
        )

    x, y, w, h = map(
        float,
        annotation["bbox"]
    )

    metadata = image_lookup[
        annotation["image_id"]
    ]

    if (
        not np.all(np.isfinite([x, y, w, h]))
        or x < 0
        or y < 0
        or w <= 0
        or h <= 0
        or x + w > metadata["width"] + 0.01
        or y + h > metadata["height"] + 0.01
    ):

        raise ValueError(
            f"Invalid annotation: {annotation['id']}"
        )

    ground_truth[
        annotation["image_id"]
    ].append(
        np.array([
            x, y, x + w, y + h
        ])
    )

total_gt = sum(
    len(ground_truth[image["id"]])
    for image in images
)

if total_gt == 0:

    raise ValueError(
        "No ground-truth faces found."
    )

print("\n========================================")
print("SCRFD 10G - AUTOMATIC OPTIMIZATION")
print("========================================")

print("Number of images:", len(images))
print("Ground-truth faces:", total_gt)


# ==========================================
# LOAD EVALUATION IMAGES
# ==========================================

image_cache = {}

for image_info in images:

    image_id = image_info["id"]

    filename = Path(
        image_info["file_name"]
    ).name

    path = IMAGE_DIR / filename

    if not path.exists():

        raise FileNotFoundError(path)

    with Image.open(path) as source:

        rgb = np.asarray(
            source.convert("RGB")
        )

    h, w = rgb.shape[:2]

    if (
        w != image_info["width"]
        or h != image_info["height"]
    ):

        raise ValueError(
            f"Image dimensions do not match: {filename}"
        )

    image_cache[image_id] = rgb


# ==========================================
# INITIALIZE MODEL
# ==========================================

detector = SCRFDDetector()

dummy = np.zeros(
    (640, 640, 3),
    dtype=np.uint8
)

for _ in range(3):

    detector.detect_raw(dummy)


# ==========================================
# IOU MATCHING
# ==========================================

def match_predictions(
    predictions,
    actual
):

    used = set()

    correct = []
    incorrect = []
    missed = []

    for box, score in sorted(
        predictions,
        key=lambda item: item[1],
        reverse=True
    ):

        best_iou = 0
        best_index = -1

        for i, gt in enumerate(actual):

            if i in used:
                continue

            iou = calculate_iou(
                box, gt
            )

            if iou > best_iou:

                best_iou = iou
                best_index = i

        if (
            best_index >= 0
            and best_iou >= IOU_THRESHOLD
        ):

            used.add(best_index)

            correct.append((
                box,
                score,
                best_iou,
                actual[best_index]
            ))

        else:

            incorrect.append((
                box,
                score
            ))

    for i, box in enumerate(actual):

        if i not in used:

            missed.append(box)

    return (
        correct,
        incorrect,
        missed
    )


# ==========================================
# CALCULATE METRICS
# ==========================================

def evaluate(predictions, threshold):

    tp = 0
    fp = 0
    fn = 0

    ious = []

    for image_info in images:

        image_id = image_info["id"]

        selected = [

            (box, score)

            for box, score in predictions[image_id]

            if score >= threshold
        ]

        correct, incorrect, missed = (
            match_predictions(
                selected,
                ground_truth[image_id]
            )
        )

        tp += len(correct)
        fp += len(incorrect)
        fn += len(missed)

        ious.extend(
            item[2] for item in correct
        )

    precision = (
        tp / (tp + fp)
        if tp + fp else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)

        if precision + recall else 0
    )

    return {
        "confidence": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_percent": precision * 100,
        "recall_percent": recall * 100,
        "f1_percent": f1 * 100,

        "mean_matched_iou": (
            float(np.mean(ious))
            if ious else 0
        )
    }


# ==========================================
# AUTOMATIC EXPERIMENTS
# ==========================================

all_results = []

best_result = None
best_predictions = None

experiment_summaries = []

for experiment in EXPERIMENTS:

    name = experiment["name"]

    print(
        f"\n========== {name} =========="
    )

    raw_cache = {}

    raw_times = []

    detector.inference_count = 0
    detector.inference_time = 0.0

    for index, image_info in enumerate(
        images,
        1
    ):

        image_id = image_info["id"]

        rgb = image_cache[image_id]

        start = time.perf_counter()

        raw = detector.detect_raw(
            rgb,

            rgb_input=experiment["rgb"],

            clahe=experiment["clahe"],

            minimum_confidence=MIN_CONFIDENCE
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        raw_times.append(elapsed)

        raw_cache[image_id] = raw

        print(
            f"[{index}/{len(images)}] "
            f"{image_info['file_name']} | "
            f"{elapsed:.2f} ms"
        )

    mean_inference_ms = (
        detector.inference_time * 1000
        / max(detector.inference_count, 1)
    )

    experiment_best = None

    for nms_threshold in NMS_THRESHOLDS:

        predictions = {}

        nms_times = []

        for image_id, raw in raw_cache.items():

            t0 = time.perf_counter()

            predictions[image_id] = (
                non_max_suppression(
                    raw,
                    nms_threshold
                )
            )

            nms_times.append(
                (time.perf_counter() - t0) * 1000
            )

        average_detection_ms = float(
            np.mean(raw_times)
            + np.mean(nms_times)
        )

        for threshold in CONF_THRESHOLDS:

            result = evaluate(
                predictions,
                threshold
            )

            result.update({

                "experiment": name,

                "rgb_input": experiment["rgb"],

                "clahe": experiment["clahe"],

                "nms_threshold": nms_threshold,

                "average_detection_ms": (
                    average_detection_ms
                ),

                "average_inference_ms": (
                    mean_inference_ms
                )
            })

            all_results.append(
                result.copy()
            )

            rank = (
                result["f1_percent"],
                result["precision_percent"]
            )

            if experiment_best is None or rank > (
                experiment_best["f1_percent"],
                experiment_best["precision_percent"]
            ):

                experiment_best = result.copy()

            if best_result is None or rank > (
                best_result["f1_percent"],
                best_result["precision_percent"]
            ):

                best_result = result.copy()

                best_predictions = predictions

    experiment_summaries.append(
        experiment_best
    )

    print(
        f"\nBest configuration for {name}"
    )

    print(
        f"Confidence: "
        f"{experiment_best['confidence']}"
    )

    print(
        f"NMS: "
        f"{experiment_best['nms_threshold']}"
    )

    print(
        f"Precision: "
        f"{experiment_best['precision_percent']:.2f}%"
    )

    print(
        f"Recall: "
        f"{experiment_best['recall_percent']:.2f}%"
    )

    print(
        f"F1: "
        f"{experiment_best['f1_percent']:.2f}%"
    )


# ==========================================
# BEST CONFIGURATION
# ==========================================

best_confidence = best_result["confidence"]

print("\n========== BEST CONFIGURATION ==========")

print("Experiment:", best_result["experiment"])
print("RGB:", best_result["rgb_input"])
print("CLAHE:", best_result["clahe"])
print("Confidence:", best_confidence)
print("NMS:", best_result["nms_threshold"])


# ==========================================
# COCO AP METRICS
# ==========================================

coco_predictions = []

for image_info in images:

    image_id = image_info["id"]

    for box, score in best_predictions[image_id]:

        x1, y1, x2, y2 = map(
            float, box
        )

        coco_predictions.append({

            "image_id": image_id,

            "category_id": category_id,

            "bbox": [
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ],

            "score": float(score)
        })

ap50 = 0
ap75 = 0
ap50_95 = 0

if coco_predictions:

    coco_dt = coco.loadRes(
        coco_predictions
    )

    evaluator = COCOeval(
        coco,
        coco_dt,
        "bbox"
    )

    evaluator.params.catIds = [
        category_id
    ]

    evaluator.params.imgIds = [
        image["id"] for image in images
    ]

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    ap50_95 = float(
        evaluator.stats[0] * 100
    )

    ap50 = float(
        evaluator.stats[1] * 100
    )

    ap75 = float(
        evaluator.stats[2] * 100
    )


# ==========================================
# ERROR ANALYSIS
# ==========================================

def draw_box(
    draw,
    box,
    color,
    label
):

    x1, y1, x2, y2 = map(
        int,
        np.round(box)
    )

    draw.rectangle(
        (x1, y1, x2, y2),
        outline=color,
        width=2
    )

    draw.text(
        (x1, max(0, y1 - 14)),
        label,
        fill=color,
        stroke_width=1,
        stroke_fill="black"
    )


size_results = {
    "small_under_32px": {
        "total": 0,
        "matched": 0
    },
    "medium_32_to_95px": {
        "total": 0,
        "matched": 0
    },
    "large_96px_and_above": {
        "total": 0,
        "matched": 0
    }
}


def get_size_group(box):

    size = np.sqrt(
        max(0, box[2] - box[0])
        * max(0, box[3] - box[1])
    )

    if size < 32:

        return "small_under_32px"

    if size < 96:

        return "medium_32_to_95px"

    return "large_96px_and_above"


frame_results = []

localization_errors = 0

for image_info in images:

    image_id = image_info["id"]

    filename = Path(
        image_info["file_name"]
    ).name

    actual = ground_truth[image_id]

    selected = [
        (box, score)

        for box, score in best_predictions[image_id]

        if score >= best_confidence
    ]

    correct, incorrect, missed = (
        match_predictions(
            selected,
            actual
        )
    )

    image = Image.fromarray(
        image_cache[image_id].copy()
    )

    draw = ImageDraw.Draw(image)

    for gt in actual:

        size_results[
            get_size_group(gt)
        ]["total"] += 1

        draw_box(
            draw,
            gt,
            "#00ccff",
            "GT"
        )

    for box, score, iou, gt in correct:

        size_results[
            get_size_group(gt)
        ]["matched"] += 1

        draw_box(
            draw,
            box,
            "#00ff00",
            f"TP {score:.2f} IoU:{iou:.2f}"
        )

    for box, score in incorrect:

        draw_box(
            draw,
            box,
            "#ff0000",
            f"FP {score:.2f}"
        )

    for box in missed:

        nearest_iou = max(
            (
                calculate_iou(box, pred)
                for pred, score in selected
            ),
            default=0
        )

        if 0.10 <= nearest_iou < IOU_THRESHOLD:

            localization_errors += 1

            label = (
                f"FN alignment: {nearest_iou:.2f}"
            )

        else:

            label = "FN missed"

        draw_box(
            draw,
            box,
            "#ffff00",
            label
        )

    image.save(
        VISUAL_DIR / filename
    )

    frame_results.append({

        "image": filename,

        "ground_truth": len(actual),

        "true_positives": len(correct),

        "false_positives": len(incorrect),

        "false_negatives": len(missed)
    })

for group, result in size_results.items():

    total = result["total"]

    result["recall_percent"] = (

        round(
            result["matched"] / total * 100,
            2
        )

        if total else None
    )


# ==========================================
# FINAL REPORT
# ==========================================

report = {

    "model": "SCRFD 10G",

    "images_tested": len(images),

    "ground_truth_faces": total_gt,

    "best_experiment": best_result["experiment"],

    "rgb_input": best_result["rgb_input"],

    "clahe_enabled": best_result["clahe"],

    "best_confidence": best_confidence,

    "best_nms_threshold": best_result["nms_threshold"],

    "iou_threshold": IOU_THRESHOLD,

    "minimum_prediction_confidence": MIN_CONFIDENCE,

    "true_positives": best_result["true_positives"],

    "false_positives": best_result["false_positives"],

    "false_negatives": best_result["false_negatives"],

    "precision_percent": round(
        best_result["precision_percent"], 2
    ),

    "recall_percent": round(
        best_result["recall_percent"], 2
    ),

    "f1_percent": round(
        best_result["f1_percent"], 2
    ),

    "AP50_percent": round(ap50, 2),

    "AP75_percent": round(ap75, 2),

    "AP50_95_percent": round(ap50_95, 2),

    "mean_matched_iou": round(
        best_result["mean_matched_iou"], 4
    ),

    "possible_localization_errors": (
        localization_errors
    ),

    "average_detection_ms": round(
        best_result["average_detection_ms"], 2
    ),

    "average_inference_ms": round(
        best_result["average_inference_ms"], 2
    ),

    "detector_only_fps": round(
        1000 / max(
            best_result["average_detection_ms"],
            1e-6
        ),
        2
    ),

    "configurations_tested": len(all_results),

    "face_size_analysis": size_results
}


# ==========================================
# SAVE REPORTS
# ==========================================

with open(
    OUTPUT_DIR / "accuracy_report.json",
    "w"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )

with open(
    OUTPUT_DIR / "predictions.json",
    "w"
) as file:

    json.dump(
        coco_predictions,
        file,
        indent=2
    )

with open(
    OUTPUT_DIR / "best_configuration.json",
    "w"
) as file:

    json.dump({

        "USE_RGB_INPUT": best_result["rgb_input"],

        "ENABLE_CLAHE": best_result["clahe"],

        "HIGH_CONF": best_confidence,

        "NMS_THRESHOLD": best_result["nms_threshold"]

    }, file, indent=4)


def save_csv(path, rows):

    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


save_csv(
    OUTPUT_DIR / "all_configurations.csv",
    all_results
)

save_csv(
    OUTPUT_DIR / "experiment_comparison.csv",
    experiment_summaries
)

save_csv(
    OUTPUT_DIR / "frame_results.csv",
    frame_results
)


# ==========================================
# ACCURACY GRAPH
# ==========================================

labels = [
    "Precision",
    "Recall",
    "F1",
    "AP50",
    "AP75",
    "AP50-95"
]

values = [
    report["precision_percent"],
    report["recall_percent"],
    report["f1_percent"],
    ap50,
    ap75,
    ap50_95
]

plt.figure(figsize=(11, 6))

bars = plt.bar(
    labels,
    values
)

plt.bar_label(
    bars,
    fmt="%.2f%%",
    padding=3
)

plt.axhline(
    90,
    linestyle="--",
    label="Target: 90%"
)

plt.ylim(0, 110)

plt.ylabel("Percentage (%)")

plt.title(
    "SCRFD 10G - Optimized Face Detection"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "accuracy_metrics.png",
    dpi=200
)

plt.close()


# ==========================================
# MODEL COMPARISON GRAPH
# ==========================================

labels = [
    "Precision",
    "Recall",
    "F1",
    "AP50"
]

previous = [
    80.00,
    57.69,
    67.04,
    54.84
]

current = [
    report["precision_percent"],
    report["recall_percent"],
    report["f1_percent"],
    report["AP50_percent"]
]

x = np.arange(len(labels))

width = 0.35

plt.figure(figsize=(11, 6))

bars1 = plt.bar(
    x - width / 2,
    previous,
    width,
    label="SCRFD 2.5G"
)

bars2 = plt.bar(
    x + width / 2,
    current,
    width,
    label="SCRFD 10G"
)

plt.bar_label(
    bars1,
    fmt="%.1f",
    padding=3
)

plt.bar_label(
    bars2,
    fmt="%.1f",
    padding=3
)

plt.xticks(x, labels)

plt.ylim(0, 110)

plt.ylabel("Percentage (%)")

plt.title(
    "SCRFD 2.5G vs SCRFD 10G"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison.png",
    dpi=200
)

plt.close()


# ==========================================
# PREPROCESSING COMPARISON GRAPH
# ==========================================

names = [
    result["experiment"]
    for result in experiment_summaries
]

precision_values = [
    result["precision_percent"]
    for result in experiment_summaries
]

recall_values = [
    result["recall_percent"]
    for result in experiment_summaries
]

f1_values = [
    result["f1_percent"]
    for result in experiment_summaries
]

x = np.arange(len(names))
width = 0.25

plt.figure(figsize=(11, 6))

plt.bar(
    x - width,
    precision_values,
    width,
    label="Precision"
)

plt.bar(
    x,
    recall_values,
    width,
    label="Recall"
)

plt.bar(
    x + width,
    f1_values,
    width,
    label="F1"
)

plt.xticks(x, names)

plt.ylim(0, 110)

plt.ylabel("Percentage (%)")

plt.title(
    "RGB vs BGR vs CLAHE"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "preprocessing_comparison.png",
    dpi=200
)

plt.close()


# ==========================================
# PRINT FINAL RESULTS
# ==========================================

print("\n========== FINAL RESULTS ==========")

for key, value in report.items():

    print(f"{key}: {value}")

print("\n========== BASELINE IMPROVEMENT ==========")

for name, old, new in zip(
    labels,
    previous,
    current
):

    print(
        f"{name}: "
        f"{old:.2f}% -> "
        f"{new:.2f}% "
        f"({new - old:+.2f} percentage points)"
    )

print("\n========== FACE SIZE ANALYSIS ==========")

for group, result in size_results.items():

    print(group, result)

print("\nResults directory:")

print(OUTPUT_DIR)

print("\nBest configuration saved successfully.")

print("\nEvaluation completed.")
