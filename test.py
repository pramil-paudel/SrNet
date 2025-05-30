"""This module is used to test the SrNet model."""
import os
import matplotlib
import numpy as np
import torch
from glob import glob
import pickle

from model.model import Srnet

# Set the backend to Agg for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from skimage import io
import pickle

# === Config ===
TEST_BATCH_SIZE = 40
COVER_PATH = "/scratch/p522p287/DATA/STEN_DATA/COCO_OUT/cover_test"
STEGO_PATH = "/scratch/p522p287/DATA/STEN_DATA/COCO_OUT/container_test"
CHKPT = "/scratch/p522p287/CODE/SrNet/checkpoints/net_100.pt"

# === Collect file paths ===
cover_image_names = sorted(glob(f"{COVER_PATH}/*.pgm"))
stego_image_names = sorted(glob(f"{STEGO_PATH}/*.pgm"))

assert len(cover_image_names) == len(stego_image_names), "Mismatched number of cover and stego images."

# === Load model ===
model = Srnet().cuda()
ckpt = torch.load(CHKPT, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# === Metrics tracking ===
test_accuracy = []
all_labels = []
all_probs = []

# === Batch-wise testing ===
for idx in range(0, len(cover_image_names), TEST_BATCH_SIZE // 2):
    cover_batch = cover_image_names[idx: idx + TEST_BATCH_SIZE // 2]
    stego_batch = stego_image_names[idx: idx + TEST_BATCH_SIZE // 2]

    batch = []
    batch_labels = []
    xi = 0
    yi = 0
    for i in range(2 * len(cover_batch)):
        if i % 2 == 0:
            batch.append(stego_batch[xi])
            batch_labels.append(1)
            xi += 1
        else:
            batch.append(cover_batch[yi])
            batch_labels.append(0)
            yi += 1

    images = torch.zeros((TEST_BATCH_SIZE, 1, 128, 128), dtype=torch.float).cuda()

    for i in range(TEST_BATCH_SIZE):
        image_path = batch[i]
        try:
            image = io.imread(image_path)
            if len(image.shape) == 3:
                image = np.mean(image, axis=2)
            image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
            images[i, 0, :, :] = torch.tensor(image).cuda()
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            continue

    with torch.no_grad():
        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long).cuda()
        outputs = model(images)
        predictions = outputs.data.max(1)[1]
        accuracy = (
            predictions.eq(batch_labels_tensor).sum() * 100.0 / batch_labels_tensor.size()[0]
        )
        test_accuracy.append(accuracy.item())

        probs = torch.softmax(outputs, dim=1)
        all_labels.append(batch_labels_tensor.cpu().numpy())
        all_probs.append(probs[:, 1].detach().cpu().numpy())

# === Flatten and prepare for ROC ===
if len(all_labels) == 0 or len(all_probs) == 0:
    print("No data available for ROC calculation.")
else:
    all_labels = np.concatenate(all_labels).ravel()
    all_probs = np.concatenate(all_probs).ravel()

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # === Save ROC data ===
    roc_data = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
        "labels": all_labels,
        "probs": all_probs
    }
    with open("roc_data.pkl", "wb") as f:
        pickle.dump(roc_data, f)
    print("✅ ROC data saved to roc_data.pkl")

    # === Plot ROC (black & white, publication-ready) ===
    roc_data = {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc
    }

    with open("roc_data.pkl", "wb") as f:
        pickle.dump(roc_data, f)

    print("✅ ROC data saved as 'roc_data.pkl'")

    plt.figure(figsize=(6, 5))
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    plt.plot(fpr, tpr, color='black', linestyle='-', linewidth=2.5, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("roc_curve_paper.png", dpi=300)
    print("📊 ROC curve saved as roc_curve_paper.png")

# === Print final accuracy ===
if len(test_accuracy) > 0:
    print(f"🧪 Average test accuracy = {sum(test_accuracy)/len(test_accuracy):.2f}%")
else:
    print("No test accuracy available.")