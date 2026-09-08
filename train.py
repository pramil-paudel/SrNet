"""This module is used to train the Srnet model.

Revised for diagnostic use. Changes from the original, with reasons:

  1. REMOVED the duplicate optimizer.step(). Adamax advances its moment
     estimates on every call, so stepping twice applied a second update
     from stale gradients and corrupted optimizer state each iteration.
     This is the most likely cause of loss never leaving ln(2)=0.69315.

  2. REPLACED RandomRotation(degrees=90). A scalar `degrees` means a
     random angle in [-90, +90], NOT multiples of 90 -- arbitrary-angle
     rotation resamples the pixel grid and damages the exact local pixel
     relationships SRNet's residual layers depend on. Now uses only
     lossless transforms (90-degree multiples and flips), which are pure
     index permutations. AUGMENT=False disables augmentation entirely so
     train and validation see identical processing -- recommended for a
     first diagnostic run.

  3. ADDED dataset size reporting at startup, so a one-batch-per-epoch
     data loading problem is visible immediately rather than inferred
     later from metric granularity.

  4. ADDED per-epoch CSV logging of train/valid loss and accuracy, so
     convergence curves can be plotted directly.

  5. ADDED FRESH_START to ignore existing checkpoints, preventing a
     silent resume from a previous run's corrupted optimizer state.

  6. ADDED an overfit check. On a small dataset a WORKING detector must
     be able to drive training accuracy well above chance. If training
     accuracy stays at ~50% on a few hundred pairs, the problem is
     mechanical, not a matter of dataset scale.

The model, loss, and optimizer choice are unchanged.
"""

import csv
import logging
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import dataset
from opts.options import arguments
from model.model import Srnet
from utils.utils import (
    latest_checkpoint,
    adjust_learning_rate,
    weights_init,
    saver,
)

# ──────────────────────────────────────────────────────────
#  Config — paths and hyperparameters set directly here
# ──────────────────────────────────────────────────────────
# Dataset produced by generate_steganalysis_dataset.py. Its layout is
#   DATA_ROOT/{train,val,test}/{cover,stego}/NNNNNN.png
# with filenames corresponding one-to-one across cover/ and stego/.
DATA_ROOT = "/home/p522p287/scratch/DATA/STEN_DATA_LENSLESS/STEGANALYSIS/imagenet_diffhide_amp4x/"

COVER_PATH       = os.path.join(DATA_ROOT, "train", "cover")
STEGO_PATH       = os.path.join(DATA_ROOT, "train", "stego")
VALID_COVER_PATH = os.path.join(DATA_ROOT, "val", "cover")
VALID_STEGO_PATH = os.path.join(DATA_ROOT, "val", "stego")

# Number of PAIRS in each split. Must not exceed what is on disk --
# set to None to count the files automatically.
TRAIN_SIZE = None
VAL_SIZE   = None

CHECKPOINTS_DIR = "./checkpoints/"

# ---- Training ----
BATCH_SIZE = 8
LR         = 1e-3
NUM_EPOCHS = 100

# ---- Diagnostics ----
# Ignore any existing checkpoints and start from scratch. Set True for
# the diagnostic rerun so a corrupted optimizer state from the previous
# run cannot be silently resumed.
FRESH_START = True

# Lossless augmentation only (90-degree rotations + flips). Set False to
# disable augmentation entirely, matching the validation transform --
# recommended for the first run so train/val processing is identical.
AUGMENT = False

# Per-epoch curve log, for plotting convergence in the write-up.
CURVE_CSV = "training_curves.csv"

# On a small dataset, a working detector should overfit. Warn if training
# accuracy has not exceeded this by CHECK_EPOCH.
OVERFIT_TARGET_ACC = 70.0
CHECK_EPOCH = 20
# ──────────────────────────────────────────────────────────

opt = arguments()

# Override the CLI/defaults with the config block above, so this script is
# self-contained and does not depend on opts/options.py being edited.
def _count_pairs(d):
    """Number of images in a split directory, used when *_SIZE is None."""
    if not os.path.isdir(d):
        sys.exit(f"Dataset directory not found: {d}")
    n = len([f for f in os.listdir(d) if f.lower().endswith((".png", ".pgm", ".jpg", ".jpeg"))])
    if n == 0:
        sys.exit(f"No images found in: {d}")
    return n

opt.cover_path       = COVER_PATH
opt.stego_path       = STEGO_PATH
opt.valid_cover_path = VALID_COVER_PATH
opt.valid_stego_path = VALID_STEGO_PATH
opt.checkpoints_dir  = CHECKPOINTS_DIR
opt.batch_size       = BATCH_SIZE
opt.lr               = LR
opt.num_epochs       = NUM_EPOCHS

# Counting from disk avoids the silent failure mode where *_SIZE exceeds
# the number of files present and the loader yields fewer pairs than
# intended -- or indexes past the end.
opt.train_size = TRAIN_SIZE if TRAIN_SIZE is not None else _count_pairs(COVER_PATH)
opt.val_size   = VAL_SIZE   if VAL_SIZE   is not None else _count_pairs(VALID_COVER_PATH)

# Cover and stego counts must match, or pairing is broken.
for _c, _s, _name in ((COVER_PATH, STEGO_PATH, "train"),
                      (VALID_COVER_PATH, VALID_STEGO_PATH, "val")):
    _nc, _ns = _count_pairs(_c), _count_pairs(_s)
    if _nc != _ns:
        sys.exit(f"{_name}: cover has {_nc} images but stego has {_ns}. "
                 "Pairing would be misaligned -- regenerate the dataset.")

logging.basicConfig(
    filename="training.log",
    format="%(asctime)s %(message)s",
    level=logging.DEBUG,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_train_transform(augment: bool):
    """
    Steganalysis augmentation must be LOSSLESS. Only 90-degree rotations
    and mirror flips qualify: they permute pixel positions without
    resampling, so the embedding residual survives intact. Any other
    angle interpolates and destroys the signal being detected.
    """
    if not augment:
        return transforms.ToTensor()

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomChoice([
            transforms.Lambda(lambda im: im),
            transforms.Lambda(lambda im: im.transpose(Image.ROTATE_90)),
            transforms.Lambda(lambda im: im.transpose(Image.ROTATE_180)),
            transforms.Lambda(lambda im: im.transpose(Image.ROTATE_270)),
        ]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


def init_curve_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "valid_loss",
                 "train_acc", "valid_acc", "lr", "epoch_seconds"]
            )


def append_curve_csv(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


if __name__ == "__main__":

    train_data = dataset.DatasetLoad(
        opt.cover_path,
        opt.stego_path,
        opt.train_size,
        transform=build_train_transform(AUGMENT),
    )

    val_data = dataset.DatasetLoad(
        opt.valid_cover_path,
        opt.valid_stego_path,
        opt.val_size,
        transform=transforms.ToTensor(),
    )

    # Creating training and validation loader.
    train_loader = DataLoader(
        train_data, batch_size=opt.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        val_data, batch_size=opt.batch_size, shuffle=False
    )

    # ---- report what the loaders actually see ----
    # A single batch per epoch means metrics are computed over one batch
    # and the network sees a fraction of the intended data.
    print("=" * 66)
    print("DATA")
    print(f"  root          : {DATA_ROOT}")
    print(f"  train cover   : {opt.cover_path}")
    print(f"  train stego   : {opt.stego_path}")
    print(f"  val cover     : {opt.valid_cover_path}")
    print(f"  val stego     : {opt.valid_stego_path}")
    print(f"  train_size    : {opt.train_size}   val_size: {opt.val_size}")
    print(f"  train pairs   : {len(train_data)}   batches: {len(train_loader)}")
    print(f"  val pairs     : {len(val_data)}   batches: {len(valid_loader)}")
    print(f"  batch size    : {opt.batch_size}  (each batch yields {2*opt.batch_size} images: cover+stego)")
    print(f"  augmentation  : {'lossless 90-deg rotations + flips' if AUGMENT else 'DISABLED'}")
    print(f"  learning rate : {opt.lr}")
    print(f"  epochs        : {opt.num_epochs}")
    print("=" * 66)

    if len(train_loader) <= 1:
        print("\n[WARN] Only one training batch per epoch. Check opt.train_size")
        print("       and the dataset paths -- the network is seeing a fraction")
        print("       of the intended data.\n")

    # model creation and initialization.
    model = Srnet()
    model = model.apply(weights_init)
    model.to(device)

    # Loss function and Optimizer
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=opt.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )

    check_point = None if FRESH_START else latest_checkpoint()
    if not check_point:
        START_EPOCH = 1
        if not os.path.exists(opt.checkpoints_dir):
            os.makedirs(opt.checkpoints_dir)
        if FRESH_START:
            print("FRESH_START=True -- ignoring any existing checkpoints.")
        else:
            print("No checkpoints found!!, Retraining started... ")
    else:
        pth = opt.checkpoints_dir + "net_" + str(check_point) + ".pt"
        ckpt = torch.load(pth)
        START_EPOCH = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        print("Model Loaded from epoch " + str(START_EPOCH) + "..")

    init_curve_csv(CURVE_CSV)
    best_train_acc = 0.0

    print(f"\nchance-level loss (ln 2) = {np.log(2):.5f}")
    print("training loss must fall clearly below this for learning to be happening\n")

    for epoch in range(START_EPOCH, opt.num_epochs + 1):
        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []

        # Training
        model.train()
        st_time = time.time()
        adjust_learning_rate(optimizer, epoch)

        for i, train_batch in enumerate(train_loader):
            # Move individual components of the dictionary to the device
            cover = train_batch["cover"].to(device)
            stego = train_batch["stego"].to(device)
            label_0 = train_batch["label"][0].to(device)
            label_1 = train_batch["label"][1].to(device)

            # Concatenate the cover and stego images, as well as the labels
            images = torch.cat((cover, stego), 0)
            labels = torch.cat((label_0, label_1), 0)

            # Ensure images and labels are in the correct data types
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # NOTE: the original called optimizer.step() a second time here.
            # Removed -- see module docstring.

            training_loss.append(loss.item())
            prediction = outputs.data.max(1)[1]
            accuracy = (
                prediction.eq(labels.data).sum() * 100.0 / (labels.size()[0])
            )
            training_accuracy.append(accuracy.item())

            sys.stdout.write(
                f"\r Epoch:{epoch}/{opt.num_epochs}"
                f" Batch:{i+1}/{len(train_loader)}"
                f" Loss:{training_loss[-1]:.4f}"
                f" Acc:{training_accuracy[-1]:.2f}"
                f" LR:{optimizer.param_groups[0]['lr']:.6f}"
            )

        end_time = time.time()

        # Validation
        model.eval()
        with torch.no_grad():

            for i, val_batch in enumerate(valid_loader):
                images = torch.cat((val_batch["cover"], val_batch["stego"]), 0)
                labels = torch.cat(
                    (val_batch["label"][0], val_batch["label"][1]), 0
                )

                images = images.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                outputs = model(images)

                loss = loss_fn(outputs, labels)
                validation_loss.append(loss.item())
                prediction = outputs.data.max(1)[1]
                accuracy = (
                    prediction.eq(labels.data).sum()
                    * 100.0
                    / (labels.size()[0])
                )
                validation_accuracy.append(accuracy.item())

        avg_train_loss = sum(training_loss) / len(training_loss)
        avg_valid_loss = sum(validation_loss) / len(validation_loss)
        avg_train_acc = sum(training_accuracy) / len(training_accuracy)
        avg_valid_acc = sum(validation_accuracy) / len(validation_accuracy)
        epoch_seconds = end_time - st_time

        best_train_acc = max(best_train_acc, avg_train_acc)

        message = (
            f"Epoch: {epoch}. "
            f"Train Loss:{avg_train_loss:.5f}. "
            f"Valid Loss:{avg_valid_loss:.5f}. "
            f"Train Acc:{avg_train_acc:.2f} "
            f"Valid Acc:{avg_valid_acc:.2f} "
            f"({epoch_seconds:.1f}s)"
        )
        print("\n", message)
        logging.info(message)

        append_curve_csv(CURVE_CSV, [
            epoch, avg_train_loss, avg_valid_loss,
            avg_train_acc, avg_valid_acc,
            optimizer.param_groups[0]["lr"], epoch_seconds,
        ])

        # ---- overfit check ----
        # On a small dataset, a functioning detector should be able to fit
        # the training set even if it cannot generalize. Failure to do so
        # points at a mechanical problem rather than insufficient data.
        if epoch == CHECK_EPOCH and best_train_acc < OVERFIT_TARGET_ACC:
            print("\n" + "!" * 66)
            print(f"[DIAGNOSTIC] After {CHECK_EPOCH} epochs, best training accuracy is")
            print(f"  {best_train_acc:.2f}% -- below the {OVERFIT_TARGET_ACC:.0f}% a working detector")
            print("  should reach on a small set. The network is not fitting the")
            print("  training data at all, which indicates a mechanical problem")
            print("  rather than a data-scale one. Worth checking:")
            print(f"    - learning rate (currently {optimizer.param_groups[0]['lr']}) and adjust_learning_rate()")
            print("    - what dataset.py does with 128x128 RGB vs the model's")
            print("      Type1(1, 64) single-channel 256x256 input")
            print("    - that cover/stego pairing is correct in DatasetLoad")
            print("!" * 66 + "\n")

        state = {
            "epoch": epoch,
            "opt": opt,
            "train_loss": avg_train_loss,
            "valid_loss": avg_valid_loss,
            "train_accuracy": avg_train_acc,
            "valid_accuracy": avg_valid_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr": optimizer.param_groups[0]["lr"],
        }

        saver(state, opt.checkpoints_dir, epoch)

    print("\n" + "=" * 66)
    print("SUMMARY")
    print(f"  best training accuracy : {best_train_acc:.2f}%")
    print(f"  final train loss       : {avg_train_loss:.5f}  (chance = {np.log(2):.5f})")
    print(f"  final valid loss       : {avg_valid_loss:.5f}")
    print(f"  curves written to      : {CURVE_CSV}")
    if best_train_acc < OVERFIT_TARGET_ACC:
        print("\n  Training accuracy never rose meaningfully above chance.")
        print("  This run does NOT support any conclusion about detectability --")
        print("  the detector did not learn. Debug before interpreting AUC.")
    else:
        print("\n  Detector fit the training data. Validation behaviour is now")
        print("  meaningful: a train/valid gap indicates genuine difficulty")
        print("  generalizing to the embedding, not a broken pipeline.")
    print("=" * 66)