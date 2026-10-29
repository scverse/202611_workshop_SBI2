"""Train a small MLP autoencoder on the selected cp_measure features, with annbatch feeding the batches.

The same script runs in the workshop (``--epochs 1`` on a laptop CPU, ~1 min) and on the cluster to
convergence. Input: X[:, var["selected"]] of workshop_cells.h5ad (cells without qc_fail), clipped to |z| <= 50
(X itself is not clipped), NaN -> 0 (the DMSO median after normalization).

    python train_autoencoder.py --adata workshop_cells.h5ad --epochs 1
    python train_autoencoder.py --adata workshop_cells.h5ad --epochs 300 --patience 20 \
        --out autoencoder.pt --latent-out X_ae.npy
    python train_autoencoder.py --adata workshop_cells.h5ad --init autoencoder.pt --epochs 0 --latent-out X_ae.npy
"""

import argparse
import time

import anndata as ad
import h5py
import numpy as np
import torch
from annbatch import Loader
from torch import nn

CLIP = 50.0


class AutoEncoder(nn.Module):
    def __init__(self, n_in: int, hidden: int = 256, latent: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                     nn.Linear(hidden, n_in))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def features(adata: ad.AnnData, names: list[str] | None = None) -> np.ndarray:
    """X[:, var.selected], or X[:, names] (e.g. the pretrained checkpoint's features on another tier)."""
    cols = adata.var_names.get_indexer(names) if names is not None else np.flatnonzero(adata.var["selected"].to_numpy())
    assert (cols >= 0).all(), "features missing from adata"
    x = np.asarray(adata.X[:, cols], dtype=np.float32)
    return np.clip(np.nan_to_num(x, nan=0.0), -CLIP, CLIP)


def make_loader(x: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> Loader:
    # annbatch's in-memory path: a dense AnnData, batches of `batch_size` drawn from preloaded chunks
    return Loader(batch_size=batch_size, chunk_size=32, preload_nchunks=64, shuffle=shuffle, drop_last=shuffle,
                  to="torch", preload_to_gpu=False, rng=np.random.default_rng(seed) if shuffle else None,
                  ).add_adata(ad.AnnData(X=x))


def evaluate(model, loader, dev) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(dev)
            tot += nn.functional.mse_loss(model(x)[0], x, reduction="sum").item() / x.shape[1]
            n += x.shape[0]
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", default="workshop_small.h5ad")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--patience", type=int, default=20, help="early stopping on validation loss")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", help="start from these weights (e.g. the pretrained autoencoder.pt)")
    ap.add_argument("--out", default="autoencoder_1epoch.pt")
    ap.add_argument("--latent-out", help="write the latent of every cell (.npy)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with h5py.File(args.adata, "r") as f:  # read only X/obs/var: the crops in obsm stay on disk
        adata = ad.AnnData(**{k: ad.io.read_elem(f[k]) for k in ("X", "obs", "var")})
    ckpt = torch.load(args.init, map_location="cpu", weights_only=True) if args.init else None
    names = ckpt["features"] if ckpt else None
    x_all = features(adata, names)
    keep = ~adata.obs["qc_fail"].to_numpy()
    rng = np.random.default_rng(args.seed)
    val = rng.random(adata.n_obs) < 0.1
    x_train, x_val = x_all[keep & ~val], x_all[keep & val]
    print(f"{x_all.shape[1]} selected features; train {len(x_train)}, val {len(x_val)} cells; device {dev}")

    if ckpt:
        args.hidden, args.latent = ckpt["hidden"], ckpt["latent"]
    model = AutoEncoder(x_all.shape[1], args.hidden, args.latent).to(dev)
    if ckpt:
        model.load_state_dict(ckpt["state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(args.patience // 3, 1))
    train_loader = make_loader(x_train, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(x_val, 1024, shuffle=False, seed=args.seed)

    if ckpt:
        print(f"initial val MSE of --init weights on this data: {evaluate(model, val_loader, dev):.4f}")
    best, best_state, stale, history = np.inf, None, 0, []
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        for batch in train_loader:
            x = batch["X"].to(dev)
            loss = nn.functional.mse_loss(model(x)[0], x)
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = evaluate(model, val_loader, dev)
        sched.step(v)
        history.append(v)
        print(f"epoch {epoch + 1}: val MSE {v:.4f} ({time.time() - t0:.1f}s)", flush=True)
        if v < best - 1e-4:
            best, stale = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop after {epoch + 1} epochs, best val MSE {best:.4f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "n_in": x_all.shape[1], "hidden": args.hidden, "latent": args.latent,
                    "features": names or adata.var_names[adata.var["selected"].to_numpy()].tolist(), "clip": CLIP,
                    "val_mse_history": history, "best_val_mse": best}, args.out)
        print(f"saved {args.out}")

    if args.latent_out:
        model.eval()
        with torch.no_grad():
            z = model.encoder(torch.from_numpy(x_all).to(dev)).cpu().numpy()
        np.save(args.latent_out, z.astype(np.float32))
        print(f"latent {z.shape} -> {args.latent_out}")


if __name__ == "__main__":
    main()
