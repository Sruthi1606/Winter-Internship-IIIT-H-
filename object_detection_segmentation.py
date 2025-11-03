# object_detection_segmentation.py
# Author: Lakshmi Sai Sruthi
# YOLOv8 Object Detection + Segmentation + Evaluation

from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# =========================
# PATH SETTINGS
# =========================
PROJECT_FOLDER = Path(r"D:\yoloproject")
IMAGE_PATH = PROJECT_FOLDER / "images" / "download.jpeg"  # your image path
OUTPUT_FOLDER = PROJECT_FOLDER / "YOLO_Outputs"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# =========================
# CHECK IMAGE
# =========================
if not IMAGE_PATH.exists():
    print(f"❌ Image not found at: {IMAGE_PATH}")
    sys.exit(1)
else:
    print(f"✅ Found image: {IMAGE_PATH}")

# =========================
# 1️⃣ OBJECT DETECTION
# =========================
print("\n🔍 Running YOLOv8 Object Detection...")
model_detect = YOLO("yolov8n.pt")

results_detect = model_detect.predict(
    source=str(IMAGE_PATH),
    save=True,
    project=str(OUTPUT_FOLDER),
    name="Detection_Results"
)
print("✅ Object detection done! Check 'YOLO_Outputs/Detection_Results'.")

# =========================
# 2️⃣ IMAGE SEGMENTATION
# =========================
print("\n🧩 Running YOLOv8 Segmentation (Persons only)...")
model_seg = YOLO("yolov8n-seg.pt")

results_seg = model_seg.predict(
    source=str(IMAGE_PATH),
    save=True,
    classes=[0],  # only 'person'
    project=str(OUTPUT_FOLDER),
    name="Segmentation_Results"
)
print("✅ Segmentation complete! Check 'YOLO_Outputs/Segmentation_Results'.")

# =========================
# 3️⃣ MODEL EVALUATION
# =========================
print("\n📊 Evaluating model performance (using coco128 demo)...")
metrics = model_detect.val(data='coco128.yaml')

print(f"\nPrecision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")
print(f"mAP@50: {metrics.box.map50:.3f}")
print(f"mAP@50-95: {metrics.box.map:.3f}")

# =========================
# 4️⃣ VISUALIZATION
# =========================
try:
    precision = metrics.box.p
    recall = metrics.box.r
    f1_scores = metrics.box.f1
    map50 = metrics.box.map50
    classes = range(len(precision))

    plt.figure(figsize=(10, 5))
    plt.plot(classes, precision, label='Precision', marker='o')
    plt.plot(classes, recall, label='Recall', marker='o')
    plt.plot(classes, f1_scores, label='F1 Score', marker='o')
    plt.xlabel('Class Index')
    plt.ylabel('Score')
    plt.title('Per-Class Metrics (Precision, Recall, F1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "class_metrics.png")
    plt.close()

    # Dummy confusion matrix (for visualization only)
    confusion = np.random.rand(10, 10)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, cmap='Blues', cbar=True)
    plt.title('Confusion Matrix (Demo)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "confusion_matrix.png")
    plt.close()

    print("🖼️ Graphs saved in:", OUTPUT_FOLDER)
except Exception as e:
    print("⚠️ Visualization skipped:", e)

print("\n🎉 All results generated successfully! Check 'YOLO_Outputs' folder.")
