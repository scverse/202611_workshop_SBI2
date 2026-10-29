# %% [markdown]
# # 1. Morphology data in AnnData
#
# In this notebook we load single-cell morphology data, look at how it is organised in an
# [AnnData](https://anndata.readthedocs.io) object, compare one cell's numbers with its image, and draw a first map of
# all cells. There is no deep learning yet; this is the table that every later notebook starts from.
#
# **The data.** The cells come from the JUMP Cell Painting consortium's TARGET2 plates. Cell Painting stains five parts
# of the cell (DNA, actin/Golgi/membrane "AGP", endoplasmic reticulum "ER", mitochondria "Mito" and RNA) and images them
# as five channels. The cells were treated with **DMSO**, the solvent the compounds are dissolved in, which serves as
# the "no treatment" control, or with one of 40 compounds that range from strongly active to inactive. They were
# imaged in five different labs ("sources") with similar spinning-disk confocal microscopes. The single-cell crops
# come from CP-BG-Bench, a single-cell benchmark built on JUMP.
#
# **Two sizes.** `TIER = "small"` (about 11,000 cells, the 40 curated compounds and DMSO) runs comfortably on a
# laptop. `TIER = "full"` (about 98,000 cells, all 302 TARGET2 perturbations from the same plates) is the same layout,
# ten times bigger.

# %%
TIER = "small"  # or "full"

# %%
from data import short_warnings

short_warnings()  # warnings stay on, printed without file paths

import time

import anndata as ad
import mantispy as mt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_cells, load_images, peak_memory_gb, plot_2d, tier_file, umap_2d

t_start = time.time()
path = tier_file(TIER)

# %% [markdown]
# ## What is in the file
#
# One file holds everything about the cells. `anndata.experimental.read_lazy` opens it without reading the data, so
# we can look at the structure first:
#
# - `obs`: one row per cell with its metadata (compound, source, plate, well, quality flags, ...).
# - `var`: one row per feature, with what the feature measures.
# - `X`: the cells x features matrix of **cp_measure** features. cp_measure is a Python re-implementation of the
#   CellProfiler measurements: size and shape of the nucleus and the cell, intensity of each channel, texture,
#   granularity, radial distribution and correlation between channels, about 2,000 numbers per cell.
# - `layers["raw"]`: the same features before normalization (small tier only).
# - `obsm`: per-cell arrays that are not features: **embeddings** (vectors a neural network computes for each cell)
#   and the cell crops themselves.

# %%
lazy = ad.experimental.read_lazy(path)
lazy

# %% [markdown]
# Now we read everything except the crops into memory. The crops stay on disk, and we read the few we need below.

# %%
cells = load_cells(path)
images, masks = load_images(path)
print(cells)
print("crops:", images.shape, images.dtype, "| masks:", masks.shape)

# %% [markdown]
# ### `obs`: the cells
#
# Columns starting with `Metadata_` follow the [mantispy](https://github.com/scverse/mantispy) schema for
# morphology data. The friendly columns (`compound`, `target`, `highlight`, `activity_class`, `is_control`) are for
# plotting. `activity_class` says whether a compound clearly changes the cells (active), a little (moderate) or not
# detectably (inactive), measured against DMSO. The `qc_*` columns flag problems; `qc_fail` is True for cells with an empty nucleus or cell mask.

# %%
cells.obs[["compound", "target", "activity_class", "Metadata_Source", "Metadata_Plate", "Metadata_Well",
           "qc_fail"]].head()

# %%
summary = cells.obs.groupby("Metadata_Source", observed=True).agg(
    cells=("compound", "size"), plates=("Metadata_Plate", "nunique"), compounds=("compound", "nunique"),
    dmso_cells=("is_control", "sum"))
summary

# %% [markdown]
# ### `var`: the features
#
# Each feature is a measurement of one object (nucleus or cell), sometimes in one channel. mantispy parsed the names
# into columns. `selected` marks the features kept by mantispy's feature selection, which removes features that are
# nearly constant, mostly missing or almost identical to another feature.

# %%
print(f"{cells.n_vars} features, {int(cells.var['selected'].sum())} selected")
cells.var[["object", "feature_group", "feature", "channel", "selected"]].sample(8, random_state=0)

# %% [markdown]
# ### `X`: normalized features
#
# Raw measurements have very different units (pixels, intensities, ratios). Each feature was therefore normalized per
# plate: subtract the plate median, divide by the spread (a robust z-score, "mad_robustize" in mantispy). A value of
# 3 means "three spreads above a typical cell on that plate". A few cells reach very large values, so we clip at
# +-50 before computing distances.

# %%
sel = cells.var["selected"].to_numpy()
X = np.clip(np.nan_to_num(cells.X[:, sel]), -50, 50)
print(cells.uns["normalization"]["method"])
pd.DataFrame(X[:, :6], columns=cells.var_names[sel][:6]).describe().round(2)

# %% [markdown]
# ## One cell, its image and its numbers
#
# Let us look at a DMSO cell and a cell treated with colchicine, which breaks down microtubules and makes cells round
# up. We read just these two crops from disk.

# %%
def show_cell(i, axes):
    img, msk = np.asarray(images[i]), np.asarray(masks[i])
    for c, ax in enumerate(axes[:5]):
        ax.imshow(img[c], cmap="gray")
        ax.set_title(cells.uns["image_channels"][c], fontsize=9)
    for m, ax in enumerate(axes[5:]):
        ax.imshow(msk[m], cmap="gray")
        ax.set_title(f"{cells.uns['mask_channels'][m]} mask", fontsize=9)
    for ax in axes:
        ax.axis("off")


obs = cells.obs.reset_index(drop=True)
ok = ~obs.qc_fail.to_numpy()
picks = {"DMSO": int(np.flatnonzero(ok & (obs.compound == "DMSO").to_numpy())[0]),
         "colchicine": int(np.flatnonzero(ok & (obs.compound == "colchicine").to_numpy())[0])}
fig, axes = plt.subplots(2, 7, figsize=(14, 4.4))
for row, (name, i) in zip(axes, picks.items()):
    show_cell(i, row)
    row[0].text(-0.3, 0.5, name, transform=row[0].transAxes, rotation=90, va="center", fontsize=11)
plt.tight_layout()

# %%
features = ["Cells_Area", "Nuclei_Area", "Cells_Eccentricity", "Cells_FormFactor",
            "Cells_Intensity_MeanIntensity__AGP", "Cells_Intensity_MeanIntensity__Mito"]
features = [f for f in features if f in cells.var_names]
cols = cells.var_names.get_indexer(features)
pd.DataFrame({name: cells.X[i, cols] for name, i in picks.items()}, index=features).round(2)

# %% [markdown]
# The numbers are z-scores on each cell's plate. The colchicine cell has a larger area and a rounder shape
# (higher form factor, lower eccentricity) than the DMSO cell, matching what we see in the images.

# %% [markdown]
# ## From cells to wells
#
# Image-based profiling usually works with one profile per **well**: the median of all cells in that well. mantispy
# does this with `tl.aggregate`.

# %%
wells = mt.tl.aggregate(cells[:, sel].copy(), by=("Metadata_Plate", "Metadata_Well"), func="median", min_cells=5)
print(wells)
wells.obs[["Metadata_Plate", "Metadata_Well", "Metadata_CellCount"]].head()

# %% [markdown]
# Aggregating throws away the differences between cells in a well: cells in different phases of the cell cycle,
# cells that respond and cells that do not. Deep learning models need many examples to learn from, and a few hundred
# wells are not enough, while tens of thousands of single cells are. That is why the single-cell table, with one row
# per cell and the crops alongside, is the starting point for the rest of the workshop.

# %% [markdown]
# ## A map of all cells
#
# To see which cells look alike, we build a **neighbor graph** (each cell connected to the 15 cells with the most
# similar features) and draw it in two dimensions with **UMAP**. Cells that are close in the plot have similar
# features.

# %%
xy = umap_2d(X)  # PCA to 50 dimensions, then UMAP on the 15-nearest-neighbor graph
plot_2d(xy, cells.obs, ["activity_class", "Metadata_Source"])
plt.tight_layout()

# %%
plot_2d(xy, cells.obs, ["highlight"], titles=["compound (curated 40 + DMSO)"], max_legend=45)
plt.tight_layout()

# %% [markdown]
# Two things stand out. Cells treated with strongly active compounds (for example the microtubule and translation
# inhibitors) move away from the DMSO cells. And cells also group by source: the same compound looks a little
# different in each lab. This **batch effect** comes back in notebook 4.

# %%
print(f"runtime {time.time() - t_start:.0f} s, peak memory {peak_memory_gb():.2f} GB")
