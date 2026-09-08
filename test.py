"""This module is used to test the Srnet model.

Revised. Bugs found in the original and what they did:

  1. NO model.eval(). BatchNorm stayed in training mode, so it normalized
     using each BATCH's statistics rather than the running statistics
     learned during training. Because every test batch was constructed
     exactly 50/50 cover/stego, batch normalization removed much of the
     very difference being detected. SRNet is BN-heavy, so this alone
     can drive AUC to 0.5.

  2. NO input scaling. Training used transforms.ToTensor(), which maps
     pixels to [0,1]. This script fed io.imread() output directly, i.e.
     [0,255] -- inputs 255x larger than the network ever saw. Activations
     saturate and the output collapses toward a constant, again giving
     AUC ~0.5 regardless of how well training went.

  3. ALTERNATING batch construction (stego, cover, stego, cover, ...).
     When scores are near-identical, ties break in list order, so each
     cover/stego pair lands adjacent in the sorted ranking. That produces
     an ROC pinned to the diagonal far more tightly than random scoring
     would -- which is exactly the anomalously straight curve seen in the
     original results.

  Also fixed: unsorted glob (cover/stego lists could be in different
  orders), a ragged final batch that indexed past the end of the list,
  and a hardcoded 256x256 allocation immediately overwritten by 128x128.

  Kept: torch.softmax over the model's LogSoftmax output. That is
  correct here -- exp(log p_i) / sum_j exp(log p_j) = p_i since the
  probabilities already sum to one.

The model itself is unchanged.
"""

import os
import sys
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from skimage import io

from model.model import Srnet


# ──────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────
DATA_ROOT  = "/lustre/i2sl/scratch/p522p287/DATA/STEGANALYSIS/imagenet_diffhide"
COVER_PATH = os.path.join(DATA_ROOT, "test", "cover")
STEGO_PATH = os.path.join(DATA_ROOT, "test", "stego")

# ---- Run identity ----
# Mirrors train_srnet.py: RUN_NAME is derived from DATA_ROOT so the test
# script automatically loads the checkpoint belonging to the dataset it is
# evaluating. Without this, every factor shared ./checkpoints/ and the
# wrong model could be loaded silently.
OUTPUT_ROOT = "./runs"
RUN_NAME    = os.path.basename(DATA_ROOT.rstrip("/"))
RUN_DIR     = os.path.join(OUTPUT_ROOT, RUN_NAME)

CHKPT      = os.path.join(RUN_DIR, "checkpoints", "net_100.pt")

IMAGE_EXT       = "png"     # "png" for the generated dataset, "pgm" for the old one
TEST_BATCH_SIZE = 40        # total images per batch (half cover, half stego)

# Match the training transform. transforms.ToTensor() divides by 255, so
# the same must happen here or the network sees a completely different
# input scale than it was trained on.
SCALE_TO_UNIT = True

# Unweighted channel mean, matching the original script. If the training
# loader converted RGB to grayscale differently (e.g. luma weights), make
# this consistent with it -- a mismatch costs real signal.
RGB_TO_GRAY = "mean"        # "mean" or "luma"

OUT_PREFIX = os.path.join(RUN_DIR, "srnet_test")
# ──────────────────────────────────────────────────────────


def load_gray(path):
    """Load an image as a 2-D float array, matching the training scale."""
    img = io.imread(path)
    if img.ndim == 3:
        if RGB_TO_GRAY == "luma":
            img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            img = np.mean(img, axis=2)
    img = img.astype(np.float32)
    if SCALE_TO_UNIT:
        img = img / 255.0
    return img


def main():
    os.makedirs(RUN_DIR, exist_ok=True)

    # sorted() so cover and stego lists correspond; bare glob() returns
    # arbitrary filesystem order and the two lists can disagree.
    cover_names = sorted(glob(f"{COVER_PATH}/*.{IMAGE_EXT}"))
    stego_names = sorted(glob(f"{STEGO_PATH}/*.{IMAGE_EXT}"))

    if not cover_names or not stego_names:
        sys.exit(f"No .{IMAGE_EXT} images found under {COVER_PATH} / {STEGO_PATH}")
    if len(cover_names) != len(stego_names):
        print(f"[WARN] {len(cover_names)} covers vs {len(stego_names)} stegos; "
              "truncating to the shorter list.")
    n_pairs = min(len(cover_names), len(stego_names))
    cover_names = cover_names[:n_pairs]
    stego_names = stego_names[:n_pairs]

    print("=" * 66)
    print(f"run        : {RUN_NAME}")
    print(f"cover      : {COVER_PATH}")
    print(f"stego      : {STEGO_PATH}")
    print(f"checkpoint : {CHKPT}")
    print(f"pairs      : {n_pairs}  ({2*n_pairs} images)")
    print(f"scaling    : {'/255 (matches ToTensor)' if SCALE_TO_UNIT else 'NONE -- will mismatch training'}")
    print("=" * 66)

    if not os.path.isfile(CHKPT):
        sys.exit(f"Checkpoint not found: {CHKPT}")

    model = Srnet().cuda()
    ckpt = torch.load(CHKPT)
    model.load_state_dict(ckpt["model_state_dict"])

    # Put BatchNorm and dropout into inference mode so running statistics
    # are used. Without this, BN normalizes each batch using that batch's
    # own statistics, which on a 50/50 cover/stego batch suppresses the
    # difference being measured.
    model.eval()

    if "epoch" in ckpt:
        print(f"loaded epoch {ckpt['epoch']}"
              + (f", valid acc {ckpt['valid_accuracy']:.2f}" if "valid_accuracy" in ckpt else ""))

    half = TEST_BATCH_SIZE // 2
    all_labels, all_probs = [], []
    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(0, n_pairs, half):
            cov_b = cover_names[start:start + half]
            stg_b = stego_names[start:start + half]
            m = min(len(cov_b), len(stg_b))
            if m == 0:
                continue
            cov_b, stg_b = cov_b[:m], stg_b[:m]

            # Grouped, not interleaved: all covers then all stegos. With
            # interleaving, tied scores break alternately and the ROC is
            # pinned artificially tight to the diagonal.
            paths = list(cov_b) + list(stg_b)
            labels = [0] * m + [1] * m

            # Allocate to the ACTUAL batch length; the original always
            # allocated TEST_BATCH_SIZE and indexed past the end of a
            # short final batch.
            first = load_gray(paths[0])
            h, w = first.shape
            images = torch.zeros((len(paths), 1, h, w), dtype=torch.float32)
            images[0, 0] = torch.from_numpy(first)
            for i, p in enumerate(paths[1:], start=1):
                images[i, 0] = torch.from_numpy(load_gray(p))

            images = images.cuda()
            labels_t = torch.tensor(labels, dtype=torch.long).cuda()

            outputs = model(images)
            pred = outputs.data.max(1)[1]
            correct += pred.eq(labels_t.data).sum().item()
            total += labels_t.size(0)

            # Model ends in LogSoftmax; softmax over log-probabilities
            # recovers the probabilities exactly, since they sum to one.
            probs = torch.softmax(outputs, dim=1)
            all_labels.append(labels_t.cpu().numpy())
            all_probs.append(probs[:, 1].detach().cpu().numpy())

    if not all_labels:
        sys.exit("No data processed.")

    y = np.concatenate(all_labels)
    p = np.concatenate(all_probs)

    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)
    accuracy = 100.0 * correct / total

    # A detector that collapsed to a constant output produces an ROC far
    # straighter than random scoring would. Comparing the maximum
    # |TPR-FPR| against the spread expected under random scoring
    # (~1.36*sqrt(2/n)) separates "learned nothing" from "output is
    # constant", which look the same in AUC alone.
    ks = float(np.max(np.abs(tpr - fpr)))
    n_per_class = len(y) // 2
    ks_expected = 1.36 * np.sqrt(2.0 / max(n_per_class, 1))
    score_std = float(np.std(p))

    print("\n" + "=" * 66)
    print("RESULTS")
    print("=" * 66)
    print(f"  accuracy        : {accuracy:.2f}%")
    print(f"  AUC             : {roc_auc:.4f}")
    print(f"  score std       : {score_std:.6f}")
    print(f"  max |TPR-FPR|   : {ks:.4f}   (random scoring would give ~{ks_expected:.4f})")

    if score_std < 1e-4:
        print("\n  [ERROR] Output scores are essentially constant. The network is")
        print("  not discriminating at all. Check that model.eval() ran, that")
        print("  inputs are scaled the same way as in training, and that the")
        print("  checkpoint loaded correctly -- do NOT read this AUC as a")
        print("  statement about the steganography.")
    elif ks < 0.3 * ks_expected:
        print("\n  [WARN] The ROC hugs the diagonal far more tightly than random")
        print("  scoring would. That points at near-constant output rather than")
        print("  a detector that genuinely learned nothing.")
    elif roc_auc < 0.55:
        print("\n  Near chance with a healthy score spread -- consistent with the")
        print("  embedding being genuinely hard to detect. Confirm with a")
        print("  positive control before treating this as a result.")
    else:
        print(f"\n  Detector separates the classes (AUC {roc_auc:.3f}).")

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"SRNet (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random guess")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title("SRNet ROC", fontsize=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_roc.png", dpi=150)

    np.savetxt(f"{OUT_PREFIX}_roc.csv",
               np.column_stack([fpr, tpr]),
               delimiter=",", header="FPR,TPR", comments="")

    np.savetxt(f"{OUT_PREFIX}_scores.csv",
               np.column_stack([y, p]),
               delimiter=",", header="label,prob_stego", comments="")

    print(f"\n  ROC plot   : {OUT_PREFIX}_roc.png")
    print(f"  ROC csv    : {OUT_PREFIX}_roc.csv")
    print(f"  raw scores : {OUT_PREFIX}_scores.csv")
    print("=" * 66)


if __name__ == "__main__":
    main()