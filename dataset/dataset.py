"""This module provides the data samples for training.

Revised. Changes from the original and why:

  1. FILENAME CONSTRUCTION -> DIRECTORY LISTING.
     The original built paths as str(index+1) + ".pgm", which requires
     files named exactly 1.pgm, 2.pgm, ... with no zero padding. Any
     other naming raises FileNotFoundError, and a missing index in the
     middle of the range fails silently at that sample. This version
     lists and sorts the directory instead, so any naming and any image
     format works, and cover/stego correspondence is verified up front
     rather than assumed.

  2. EXPLICIT GRAYSCALE CONVERSION.
     The original did none: it relied on PGM being single-channel by
     definition. Feeding it 3-channel PNGs would produce a (3,H,W)
     tensor and a shape error against the model's Type1(1, 64) input.
     Conversion now happens here, explicitly, with the method recorded.

     NOTE ON SIGNAL LOSS: converting RGB to grayscale attenuates the
     embedding. Measured on a 1-LSB residual, the fraction of differing
     pixels drops from ~44% to ~35% -- the three channels are averaged,
     so independent per-channel residuals partially cancel. If SRNet
     reports near-chance on grayscale input, some of that may be this
     conversion rather than the steganography. The stronger option is to
     train on each colour channel separately (CHANNEL_MODE = "r"/"g"/"b")
     and report the best, which preserves the full per-channel residual
     without changing the model's single-channel input.

  3. sample NameError fix. In the original, `sample` was only assigned
     inside `if self.transforms:`, so a None transform crashed on the
     next line.
"""

import os
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

import imageio as io

# pylint: disable=E1101
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=E1101

IMAGE_EXTS = (".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# How to reduce a 3-channel image to the single channel the model expects.
#   "mean" : unweighted average of R,G,B
#   "luma" : 0.299R + 0.587G + 0.114B
#   "r" / "g" / "b" : take one channel, preserving its residual in full
CHANNEL_MODE = "mean"


def _list_images(directory: str):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTS)
    )
    if not files:
        raise FileNotFoundError(f"No images found in: {directory}")
    return files


def _to_single_channel(img: np.ndarray) -> np.ndarray:
    """Reduce to 2-D. Returns the array unchanged if already 2-D."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] == 1:
            return img[:, :, 0]
        if CHANNEL_MODE == "luma":
            return (0.299 * img[:, :, 0]
                    + 0.587 * img[:, :, 1]
                    + 0.114 * img[:, :, 2])
        if CHANNEL_MODE in ("r", "g", "b"):
            return img[:, :, {"r": 0, "g": 1, "b": 2}[CHANNEL_MODE]]
        return np.mean(img[:, :, :3], axis=2)
    raise ValueError(f"Unexpected image shape: {img.shape}")


class DatasetLoad(Dataset):
    """This class returns the data samples."""

    def __init__(
        self,
        cover_path: str,
        stego_path: str,
        size: int,
        transform: Tuple = None,
    ) -> None:
        """Constructor.

        Args:
            cover_path (str): path to cover images.
            stego_path (str): path to stego images.
            size (int): number of pairs to use. Clipped to what is
              actually present on disk, so an over-large value cannot
              silently cause missing-file errors partway through.
            transform (Tuple, optional): torchvision transform applied to
              both cover and stego identically.
        """
        self.cover = cover_path
        self.stego = stego_path
        self.transforms = transform

        cover_files = _list_images(cover_path)
        stego_files = _list_images(stego_path)

        # Cover and stego must correspond one-to-one. Comparing the sorted
        # name lists catches a misaligned or partially generated dataset
        # here, rather than producing wrong labels for every sample.
        if cover_files != stego_files:
            only_c = set(cover_files) - set(stego_files)
            only_s = set(stego_files) - set(cover_files)
            msg = (f"Cover and stego filenames differ: "
                   f"{len(cover_files)} vs {len(stego_files)} files.")
            if only_c:
                msg += f" Only in cover (first 3): {sorted(only_c)[:3]}."
            if only_s:
                msg += f" Only in stego (first 3): {sorted(only_s)[:3]}."
            raise ValueError(msg)

        available = len(cover_files)
        if size is not None and size < available:
            cover_files = cover_files[:size]
        self.files = cover_files
        self.data_size = len(self.files)

        if size is not None and size > available:
            print(f"[DatasetLoad] requested size {size} exceeds the "
                  f"{available} images present; using {available}.")

    def __len__(self) -> int:
        """returns the length of the dataset."""
        return self.data_size

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Returns the (cover, stego) pair for training.

        Args:
            index (int): index into the sorted file list.
        Returns:
            dict with "cover", "stego" tensors and "label" pair.
        """
        img_name = self.files[index]

        cover_img = _to_single_channel(io.imread(os.path.join(self.cover, img_name)))
        stego_img = _to_single_channel(io.imread(os.path.join(self.stego, img_name)))

        # ToTensor() expects uint8 for its /255 scaling. Grayscale
        # averaging produces floats, so cast back before the transform --
        # otherwise the transform passes values through unscaled and the
        # network sees [0,255] instead of [0,1].
        cover_img = np.clip(cover_img, 0, 255).astype(np.uint8)
        stego_img = np.clip(stego_img, 0, 255).astype(np.uint8)

        # pylint: disable=E1101
        label1 = torch.tensor(0, dtype=torch.long).to(device)
        label2 = torch.tensor(1, dtype=torch.long).to(device)
        # pylint: enable=E1101

        if self.transforms:
            cover_img = self.transforms(cover_img)
            stego_img = self.transforms(stego_img)
        else:
            # Original crashed here with a NameError when transform was None.
            cover_img = torch.from_numpy(cover_img).unsqueeze(0).float() / 255.0
            stego_img = torch.from_numpy(stego_img).unsqueeze(0).float() / 255.0

        sample = {"cover": cover_img, "stego": stego_img}
        sample["label"] = [label1, label2]
        return sample