"""Download and load the workshop files.

Every file is fetched with pooch from the scverse example-data CDN and checked against its sha256, so a
second run uses the local copy in ./data.
"""

from __future__ import annotations

import warnings

# print warnings as "Category: message" (no file paths); set before the imports below, which can warn
warnings.formatwarning = lambda message, category, *args, **kwargs: f"{category.__name__}: {message}\n"

import platform  # noqa: E402
import resource  # noqa: E402

import anndata as ad  # noqa: E402
import h5py  # noqa: E402
import pooch  # noqa: E402

BASE_URL = "https://exampledata.scverse.org/sbi2-cytodata/workshop/"

REGISTRY = {  # sha256 of the frozen files (misc/data_manifest.sha256)
    "workshop_small.h5ad": "sha256:12a057e91b94dfe93e8aac32639debeb8d3b36bb4e022b16a75bf02d7fae637a",  # 497 MB
    "workshop_full.h5ad": "sha256:b8bb9e78d645c953bd2cebb58218c5db34f830acb678e21ceb66b6ef25821101",  # 4.3 GB
    "workshop_small_collection.zarr.zip": "sha256:f957a1f4e22f2a98d0c3f20c652e1ca7cfafa3c33dd3088f239f5662e6ae598a",  # 527 MB
    "workshop_full_collection.zarr.zip": "sha256:af1edc93ff09611b0d221272847085056b99d72b6cef726b3d6bbbafb1685343",  # 4.9 GB
    "autoencoder.pt": "sha256:fddac4f7d2b1cfc3626375414a4221b25543ba42debcf58c0d0ef91940276613",  # 2 MB
    "autoencoder_full.pt": "sha256:2a3bdc9e502e4270a585af0d7796cebf85f558c763988b3baf5c3d128931cef1",  # 2 MB
    "openphenom_demo_crops224.h5": "sha256:9500712866186ece79ec8110f2480ee9e21432d70d30b9d0a3b1e2e34ad05319",  # 39 MB
}

EMBEDDINGS = ("X_openphenom", "X_openphenom_ft", "X_ae", "X_ae_fulltrained")


def fetch(name: str) -> str:
    """Download `name` into ./data (once) and return its local path."""
    return pooch.retrieve(f"{BASE_URL}{name}", known_hash=REGISTRY[name], fname=name, path="data",
                          progressbar=True)


def tier_file(tier: str) -> str:
    return fetch(f"workshop_{tier}.h5ad")


def load_cells(path: str) -> ad.AnnData:
    """Everything except the crops, in memory: X, obs, var, layers, uns and the embeddings in obsm."""
    with h5py.File(path, "r") as f:
        return ad.AnnData(
            X=ad.io.read_elem(f["X"]),
            obs=ad.io.read_elem(f["obs"]),
            var=ad.io.read_elem(f["var"]),
            uns=ad.io.read_elem(f["uns"]),
            layers={k: ad.io.read_elem(f["layers"][k]) for k in f["layers"]},
            obsm={k: ad.io.read_elem(f["obsm"][k]) for k in EMBEDDINGS if k in f["obsm"]},
        )


def load_images(path: str):
    """The crops as lazy dask arrays (nothing is read until you index them): images (n,5,96,96), masks (n,2,96,96)."""
    f = h5py.File(path, "r")  # stays open while the arrays are in use
    chunks = (16, -1, -1, -1)
    return (ad.experimental.read_elem_lazy(f["obsm"]["single_cell_images"], chunks=chunks),
            ad.experimental.read_elem_lazy(f["obsm"]["masks"], chunks=chunks))


def peak_memory_gb() -> float:
    """Peak resident memory of this Python process so far."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if platform.system() == "Darwin" else rss / 1e6


def short_warnings() -> None:
    """Keep warnings on, but print them as 'Category: message' without the file path."""
    warnings.simplefilter("default")
    warnings.formatwarning = lambda message, category, *args, **kwargs: f"{category.__name__}: {message}\n"


def umap_2d(x, n_pcs: int = 50, n_neighbors: int = 15, seed: int = 0):
    """PCA (if wide) then UMAP to two dimensions."""
    import numpy as np
    import umap
    from sklearn.decomposition import PCA

    x = np.asarray(x, dtype=np.float32)
    if x.shape[1] > n_pcs:
        x = PCA(n_components=n_pcs, random_state=seed).fit_transform(x)
    return umap.UMAP(n_neighbors=n_neighbors, random_state=seed).fit_transform(x)


def plot_2d(xy, obs, colors, titles=None, max_legend: int = 20, axes=None):
    """Scatter a 2D embedding, one panel per obs column in `colors`."""
    import matplotlib.pyplot as plt
    import numpy as np

    if axes is None:
        _, axes = plt.subplots(1, len(colors), figsize=(6 * len(colors), 5), squeeze=False)
        axes = axes[0]
    for ax, col, title in zip(axes, colors, titles or colors):
        values = obs[col].astype(str).to_numpy()
        cats = sorted(set(values), key=lambda c: (c == "other", c))
        cmap = plt.get_cmap("tab20" if len(cats) > 10 else "tab10")
        order = np.random.default_rng(0).permutation(len(values))
        for k, c in enumerate(cats):
            m = values[order] == c
            color = "lightgrey" if c == "other" else cmap(k % cmap.N)
            ax.scatter(*xy[order][m].T, s=2, color=color, label=c, rasterized=True, linewidths=0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if len(cats) <= max_legend:
            ax.legend(markerscale=5, frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    return axes
