"""This module is used to test the Srnet model."""
from glob import glob
# from model import Srnet
from model.model import Srnet
import torch
import numpy as np
import matplotlib
 # Set the backend to Agg
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from skimage import io

TEST_BATCH_SIZE = 40
COVER_PATH = "/scratch/p522p287/DATA/STEN_DATA/IMAGE_NET_OUT/SRNET/cover_test"
STEGO_PATH = "/scratch/p522p287/DATA/STEN_DATA/IMAGE_NET_OUT/SRNET/container_test"
CHKPT = "/scratch/p522p287/CODE/SrNet/checkpoints/net_100.pt"

cover_image_names = glob(f"{COVER_PATH}/*.pgm")
stego_image_names = glob(f"{STEGO_PATH}/*.pgm")

cover_labels = np.zeros((len(cover_image_names)))
stego_labels = np.ones((len(stego_image_names)))

model = Srnet().cuda()

ckpt = torch.load(CHKPT)
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
    cover_batch = cover_image_names[idx : idx + TEST_BATCH_SIZE // 2]
    stego_batch = stego_image_names[idx : idx + TEST_BATCH_SIZE // 2]

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
    all_probs.append(probs[:, 1].cpu().numpy() if probs.ndim > 1 else probs.cpu().numpy())

# Check if all_labels or all_probs are empty
if len(all_labels) == 0 or len(all_probs) == 0:
    print("No data available for ROC calculation.")
else:
    # Flatten lists to create arrays
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Compute ROC curve and AUC for binary classification
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))  # Set figure size
    colors = sns.color_palette("husl", 1)  # Use a color palette from Seaborn
    plt.plot(fpr, tpr, color=colors[0], lw=2, label='Binary Class (AUC = {:.2f})'.format(roc_auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curve_test.png')  # Save the plot to a file

if len(test_accuracy) > 0:
    print(f"test_accuracy = {sum(test_accuracy)/len(test_accuracy):.2f}")
else:
    print("No test accuracy available.")