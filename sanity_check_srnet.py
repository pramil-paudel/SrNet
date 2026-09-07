#!/usr/bin/env python3
"""
sanity_check_srnet.py

Decisive test: can this SRNet learn a TRIVIALLY separable task?

Training loss pinned at ln(2)=0.69315 with training accuracy at ~51%
means the network is emitting a constant output. That can come from the
task being too hard OR from something mechanically broken (weight init,
learning-rate schedule, optimizer, architecture-vs-input-size). Those
demand opposite responses, and the DiffHide data cannot tell them apart.

So this trains the real Srnet on synthetic data where the two classes
differ enormously -- far beyond any steganographic residual. A working
network must reach near-100% accuracy within a few hundred iterations.

  If it LEARNS   -> model, init, loss and optimizer are fine. The problem
                    is signal strength: SRNet is being asked to find a
                    ~1 LSB residual from random init, which is exactly
                    the regime the original authors handled with a
                    curriculum (train on a strong payload first, then
                    fine-tune down). Escalate by amplifying the residual.

  If it FAILS    -> the fault is mechanical and lives in weights_init(),
                    adjust_learning_rate(), the optimizer settings, or
                    the model's behaviour at 128x128 rather than 256x256.
                    No amount of data will fix it.

Run from the SRNet repo root:  python sanity_check_srnet.py
"""

import sys

import numpy as np
import torch
from torch import nn

try:
    from model.model import Srnet
except ImportError as e:
    sys.exit(f"Run this from the SRNet repo root ({e})")

try:
    from utils.utils import weights_init, adjust_learning_rate
    HAVE_UTILS = True
except ImportError:
    HAVE_UTILS = False


# ---- settings ----
IMG_SIZE   = 128      # match your actual data
BATCH      = 8        # pairs per step -> 2*BATCH images
ITERS      = 300
LR         = 1e-3
# Difference between classes, in [0,1] units. 0.1 is ~25 LSB: enormous
# compared to the ~1 LSB DiffHide residual, and trivially separable.
DELTA      = 0.10
USE_WEIGHTS_INIT = True
USE_LR_SCHEDULE  = False   # set True to test adjust_learning_rate()


def make_batch(device):
    """Cover = uniform noise. Stego = same noise plus a large constant."""
    cover = torch.rand(BATCH, 1, IMG_SIZE, IMG_SIZE, device=device) * 0.5 + 0.25
    stego = (cover + DELTA).clamp(0, 1)
    images = torch.cat([cover, stego], 0)
    labels = torch.cat([
        torch.zeros(BATCH, dtype=torch.long),
        torch.ones(BATCH, dtype=torch.long),
    ]).to(device)
    return images, labels


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 64)
    print("SRNet sanity check -- trivially separable synthetic task")
    print("=" * 64)
    print(f"  device      : {device}")
    print(f"  image size  : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  class delta : {DELTA} in [0,1]  (~{DELTA*255:.0f} LSB)")
    print(f"  DiffHide's actual residual is ~1 LSB, i.e. {DELTA*255:.0f}x smaller")
    print(f"  weights_init: {'applied' if (USE_WEIGHTS_INIT and HAVE_UTILS) else 'NOT applied (torch default)'}")
    print(f"  lr schedule : {'adjust_learning_rate()' if (USE_LR_SCHEDULE and HAVE_UTILS) else 'fixed'}")
    print("=" * 64)

    model = Srnet().to(device)
    if USE_WEIGHTS_INIT and HAVE_UTILS:
        model = model.apply(weights_init)
    model.train()

    # quick architecture check at this input size
    with torch.no_grad():
        probe = torch.rand(2, 1, IMG_SIZE, IMG_SIZE, device=device)
        try:
            out = model(probe)
            print(f"\nforward pass OK: input {tuple(probe.shape)} -> output {tuple(out.shape)}")
            if out.shape[1] != 2:
                print(f"[ERROR] expected 2 output classes, got {out.shape[1]}")
        except Exception as exc:
            sys.exit(f"\n[ERROR] forward pass failed at {IMG_SIZE}x{IMG_SIZE}: {exc}")

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=LR,
                                    betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    chance = float(np.log(2))
    print(f"\nchance loss = {chance:.5f}\n")

    losses, accs = [], []
    for it in range(1, ITERS + 1):
        if USE_LR_SCHEDULE and HAVE_UTILS:
            adjust_learning_rate(optimizer, it)

        images, labels = make_batch(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # gradient magnitude: a zero here means nothing is reaching the weights
        gnorm = float(torch.sqrt(sum(
            p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None
        )))

        optimizer.step()

        pred = outputs.data.max(1)[1]
        acc = pred.eq(labels).float().mean().item() * 100
        losses.append(loss.item())
        accs.append(acc)

        if it % 25 == 0 or it == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  iter {it:4d}  loss {loss.item():.5f}  acc {acc:6.2f}%  "
                  f"|grad| {gnorm:.3e}  lr {lr_now:.2e}")

    last_loss = float(np.mean(losses[-25:]))
    last_acc = float(np.mean(accs[-25:]))

    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    print(f"  final loss (last 25 iters) : {last_loss:.5f}   (chance {chance:.5f})")
    print(f"  final acc  (last 25 iters) : {last_acc:.2f}%")

    if last_acc > 90:
        print("\n  LEARNS. Model, init, loss and optimizer are all functional.")
        print("  The flat curves on DiffHide data therefore reflect SIGNAL")
        print("  STRENGTH, not a mechanical fault. Next steps:")
        print("    - curriculum: train first on containers generated with a")
        print("      much larger res_scale, then fine-tune down to the real one")
        print("    - try CHANNEL_MODE='r' instead of 'mean' in dataset.py, which")
        print("      preserves one channel's residual in full instead of")
        print("      averaging three and partially cancelling them")
        print("    - more data: a few hundred pairs is very small for SRNet")
    elif last_acc > 60:
        print("\n  PARTIALLY learns. Something is impeding optimization --")
        print("  suspect the learning rate or weights_init(). Re-run this with")
        print("  USE_WEIGHTS_INIT=False and with a different LR to isolate it.")
    else:
        print("\n  DOES NOT LEARN even a trivially separable task. The fault is")
        print("  MECHANICAL, and more or better data will not help. Check, in")
        print("  order:")
        print("    - weights_init() in utils/utils.py: re-run with")
        print("      USE_WEIGHTS_INIT=False. If it learns then, that function")
        print("      is the culprit.")
        print("    - adjust_learning_rate(): print the LR it sets; a schedule")
        print("      that drives it to ~0 produces exactly this behaviour.")
        print("    - the |grad| column above: values near 0 mean no gradient")
        print("      is reaching the weights at all.")
        print("    - model behaviour at 128x128 vs the 256x256 it was written")
        print("      for -- check the pooling stack does not collapse the")
        print("      spatial dimensions.")
    print("=" * 64)


if __name__ == "__main__":
    main()
