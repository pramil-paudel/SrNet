"""This module is used to test the Srnet model."""
import matplotlib
import numpy as np
import torch
from glob import glob

# from model import Srnet
from model.model import Srnet

# Set the backend to Agg
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from skimage import io

TEST_BATCH_SIZE = 40
COVER_PATH = "/scratch/p522p287/DATA/STEN_DATA/COCO_OUT/cover_test"
STEGO_PATH = "/scratch/p522p287/DATA/STEN_DATA/COCO_OUT/container_test"
CHKPT = "/scratch/p522p287/CODE/SrNet/checkpoints/net_100.pt"

cover_image_names = glob(f"{COVER_PATH}/*.pgm")
stego_image_names = glob(f"{STEGO_PATH}/*.pgm")

cover_labels = np.zeros((len(cover_image_names)))
stego_labels = np.ones((len(stego_image_names)))

model = Srnet().cuda()

ckpt = torch.load(CHKPT, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
# pylint: disable=E1101
images = torch.empty((TEST_BATCH_SIZE, 1, 256, 256), dtype=torch.float)
# pylint: enable=E1101
test_accuracy = []
test_loss = 0
correct = 0
all_labels = []
all_probs = []
class_counts = {0: 0, 1: 0}

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
    # pylint: disable=E1101
    images = torch.zeros((TEST_BATCH_SIZE, 1, 128, 128), dtype=torch.float).cuda()
    for i in range(TEST_BATCH_SIZE):
        image_path = batch[i]
        try:
            image = io.imread(image_path)
            if len(image.shape) == 3:  # Convert RGB to grayscale if necessary
                image = np.mean(image, axis=2)
            images[i, 0, :, :] = torch.tensor(image).cuda()
        except ValueError as e:
            print(f"Error loading image {image_path}: {e}")
            continue

    image_tensor = images.cuda()
    batch_labels = torch.tensor(batch_labels, dtype=torch.long).cuda()
    # pylint: enable=E1101
    outputs = model(image_tensor)
    prediction = outputs.data.max(1)[1]

    accuracy = (
            prediction.eq(batch_labels.data).sum()
            * 100.0
            / (batch_labels.size()[0])
    )
    test_accuracy.append(accuracy.item())

    # Store labels and probabilities for ROC
    probs = torch.softmax(outputs, dim=1)  # Use softmax for class probabilities
    all_labels.append(batch_labels.cpu().numpy())
    all_probs.append(probs[:, 1].detach().cpu().numpy() if probs.ndim > 1 else probs.detach().cpu().numpy())

if len(all_labels) == 0 or len(all_probs) == 0:
    print("No data available for ROC calculation.")
else:
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # === Save ROC data as .pkl ===
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

    # === Plot: Black & White, publication quality ===
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
    plt.savefig('roc_curve_paper.png', dpi=300)

    print("📊 ROC curve saved as roc_curve_paper.png")
