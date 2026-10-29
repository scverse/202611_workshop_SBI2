"""Out-of-core proof of concept: a tiny CNN classifier trained on single-cell crops streamed from disk by annbatch.

Data: a pre-shuffled annbatch collection (zarr) of a tier; obsm/single_cell_images (n,5,96,96) uint8, downsampled on
the fly to 48 px (2x2 average pooling) and scaled to [0, 1].
Target: obs["highlight"], the 40 curated compounds + DMSO (+ "other" in the full tier); cross-entropy.
Model: 3 conv blocks (32/64/128 channels; conv + BatchNorm + ReLU + max-pool), global average pool, linear head
(~100k parameters). AdamW + OneCycleLR.
Sampling: --sampler shuffle (plain random chunks) or balanced (annbatch WeightedClassSampler, every class equally
likely; needs a collection grouped by compound, see poc_build_grouped.py).
Split: held out by plate (the same 6 of 64 plates for both tiers, seed 0); training batches drop those rows, the
validation crops are read once into memory.
Logs <out>/<tier>_<sampler>_lr<max_lr>_losses.csv: train loss/accuracy every --log-every steps, held-out loss,
accuracy and balanced accuracy per epoch. Prints images/s, wall time per epoch, the first step at which the smoothed
train loss falls below half its initial value, and peak RSS.

    python poc_annbatch_ooc.py --collection workshop_small_collection.zarr --tier small --epochs 1 --threads 4
"""

import argparse
import csv
import resource
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import zarr
from annbatch import Loader
from annbatch.samplers import WeightedClassSampler
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.nn import functional as F


class TinyCNN(nn.Module):
    def __init__(self, n_classes: int, widths=(32, 64, 128)):
        super().__init__()
        layers, c_in = [], 5
        for c in widths:
            layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(c_in, n_classes)

    def forward(self, x):
        return self.head(self.features(x))


def to_input(x: torch.Tensor) -> torch.Tensor:
    """uint8 (B, 5, 96, 96) -> float (B, 5, 48, 48) in [0, 1]."""
    return F.avg_pool2d(x.float().div_(255.0), 2)


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-lr", type=float, default=6e-3)
    ap.add_argument("--threads", type=int, default=4, help="4 mimics a laptop")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--sampler", choices=["shuffle", "balanced"], default="shuffle")
    ap.add_argument("--out", default="poc")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    run = f"{args.tier}_{args.sampler}_lr{args.max_lr:g}"

    groups = [g for _, g in sorted(zarr.open(args.collection, mode="r").groups())]
    obs = [ad.io.read_elem(g["obs"])[["Metadata_Plate", "highlight"]] for g in groups]
    images = [g["obsm"]["single_cell_images"] for g in groups]
    labels = sorted(set().union(*[set(o["highlight"].astype(str)) for o in obs]))
    label_id = {c: i for i, c in enumerate(labels)}

    # held-out plates: 10% of the sorted plate list (seed 0), identical for both tiers
    plates = np.array(sorted(set().union(*[set(o["Metadata_Plate"].astype(str)) for o in obs])))
    val_plates = set(np.random.default_rng(0).choice(plates, round(0.1 * len(plates)), replace=False))
    val_x, val_y = [], []
    for a, o in zip(images, obs):
        idx = np.flatnonzero(o["Metadata_Plate"].astype(str).isin(val_plates).to_numpy())
        val_x.append(a.oindex[idx])
        val_y.append(o["highlight"].astype(str).map(label_id).to_numpy()[idx])
    val_x, val_y = torch.from_numpy(np.concatenate(val_x)), np.concatenate(val_y)
    n_total = sum(a.shape[0] for a in images)
    n_train = n_total - len(val_x)
    print(f"{run}: {len(labels)} classes, {len(val_plates)} of {len(plates)} plates held out "
          f"({len(val_x)} of {n_total} cells), {args.threads} threads")

    if args.sampler == "shuffle":
        loader = Loader(batch_size=args.batch_size, chunk_size=16, preload_nchunks=64, shuffle=True, drop_last=True,
                        to="torch", preload_to_gpu=False, rng=np.random.default_rng(0))
    else:  # every class equally likely; a class needs contiguous runs of >= chunk_size rows (grouped collection)
        classes = pd.Categorical(pd.concat([o["highlight"].astype(str) for o in obs], ignore_index=True))
        sampler = WeightedClassSampler(chunk_size=16, preload_nchunks=64, batch_size=args.batch_size, classes=classes,
                                       num_samples=n_train // args.batch_size * args.batch_size, drop_last=True,
                                       rng=np.random.default_rng(0))
        loader = Loader(batch_sampler=sampler, to="torch", preload_to_gpu=False)
    loader.add_datasets(images, obs=obs)

    model = TinyCNN(len(labels))
    steps_per_epoch = n_train // args.batch_size
    opt = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.max_lr, total_steps=args.epochs * steps_per_epoch + 10)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e3:.0f} k parameters, ~{steps_per_epoch} steps per epoch")

    @torch.no_grad()
    def validate() -> dict[str, float]:
        model.eval()
        logits = torch.cat([model(to_input(val_x[s:s + 512])) for s in range(0, len(val_x), 512)])
        model.train()
        pred = logits.argmax(1).numpy()
        return {"val_loss": F.cross_entropy(logits, torch.from_numpy(val_y)).item(),
                "val_acc": float((pred == val_y).mean()),
                "val_balanced_acc": float(balanced_accuracy_score(val_y, pred))}

    f = open(out / f"{run}_losses.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["tier", "sampler", "max_lr", "epoch", "step", "samples_seen", "metric", "value"])
    for k, v in validate().items():
        w.writerow([args.tier, args.sampler, args.max_lr, 0, 0, 0, k, v])
    seen, step, smooth, first_loss, half_step, t_start = 0, 0, None, None, None, time.time()
    for epoch in range(1, args.epochs + 1):
        t0, n_img, losses, accs = time.time(), 0, [], []
        for batch in loader:
            keep = torch.from_numpy(~batch["obs"]["Metadata_Plate"].astype(str).isin(val_plates).to_numpy())
            if not keep.any():
                continue
            x = to_input(batch["X"][keep])
            y = torch.from_numpy(batch["obs"]["highlight"].astype(str).map(label_id).to_numpy()[keep.numpy()])
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            step += 1
            n_img += len(y)
            seen += len(y)
            losses.append(loss.item())
            accs.append((logits.argmax(1) == y).float().mean().item())
            smooth = loss.item() if smooth is None else 0.9 * smooth + 0.1 * loss.item()
            first_loss = first_loss or smooth
            if half_step is None and step > 10 and smooth < 0.5 * first_loss:
                half_step = (step, seen, time.time() - t_start)
            if step % args.log_every == 0:
                w.writerow([args.tier, args.sampler, args.max_lr, epoch, step, seen, "train_loss", np.mean(losses)])
                w.writerow([args.tier, args.sampler, args.max_lr, epoch, step, seen, "train_acc", np.mean(accs)])
                losses, accs = [], []
        dt = time.time() - t0
        v = validate()
        for k, val in v.items():
            w.writerow([args.tier, args.sampler, args.max_lr, epoch, step, seen, k, val])
        f.flush()
        print(f"[{run}] epoch {epoch}: {n_img} images in {dt:.0f} s ({n_img / dt:.0f} images/s); held-out loss "
              f"{v['val_loss']:.3f}, accuracy {v['val_acc']:.3f}, balanced accuracy {v['val_balanced_acc']:.3f}; "
              f"peak RSS {peak_rss_gb():.2f} GB", flush=True)
    f.close()
    if half_step:
        print(f"[{run}] smoothed train loss below half its initial value at step {half_step[0]} "
              f"({half_step[1]} images, {half_step[2]:.0f} s)")
    else:
        print(f"[{run}] smoothed train loss never fell below half its initial value ({first_loss:.3f} -> {smooth:.3f})")
    torch.save(model.state_dict(), out / f"{run}_cnn.pt")


if __name__ == "__main__":
    main()
