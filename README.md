# Deep learning on morphology data in the scverse ecosystem

Notebooks for the SBI2 2026 workshop by Tim Treis and Ilan Gold: single-cell morphology data in AnnData, a small model trained with annbatch, a pretrained image model on single-cell crops, and the same code scaled to many more cells.

Schedule, venue and setup: https://scverse.org/sbi2_2026/

## Setup

With [pixi](https://pixi.sh) (installs the pinned environment on first use, registers the `scverse-sbi2` kernel and starts JupyterLab in `notebooks/`):

```bash
pixi run lab
```

With conda:

```bash
conda env create -f environment.yml
conda activate scverse-sbi2
python -m ipykernel install --user --name scverse-sbi2
cd notebooks && jupyter lab
```

Everything runs on a laptop CPU with 16 GB of RAM. Each notebook downloads what it needs into `notebooks/data/` with pooch and checks the sha256 (listed in `notebooks/data.py`). To download ahead of time, save the files below into `notebooks/data/`.

## Data

Single cells from the JUMP Cell Painting TARGET2 plates, imaged in five labs with similar spinning-disk confocal microscopes, with crops from CP-BG-Bench. Two tiers share one layout; each notebook switches with `TIER = "small"` or `"full"`.

| File | Contents | Size |
|---|---|---|
| [workshop_small.h5ad](https://exampledata.scverse.org/sbi2-cytodata/workshop/workshop_small.h5ad) | 11,244 cells: DMSO and 40 curated compounds. cp_measure features in `X`, crops and masks in `obsm`, embeddings `X_openphenom`, `X_openphenom_ft`, `X_ae` | 497 MB |
| [workshop_full.h5ad](https://exampledata.scverse.org/sbi2-cytodata/workshop/workshop_full.h5ad) | 98,345 cells: all 302 TARGET2 perturbations from the same plates (the small tier is a subset) | 4.3 GB |
| [workshop_full_collection.zarr.zip](https://exampledata.scverse.org/sbi2-cytodata/workshop/workshop_full_collection.zarr.zip) | The full tier as a pre-shuffled annbatch collection (features and crops) | 4.9 GB |
| [autoencoder.pt](https://exampledata.scverse.org/sbi2-cytodata/workshop/autoencoder.pt) | Feature autoencoder trained on the small tier (gives `X_ae`) | 2 MB |
| [autoencoder_full.pt](https://exampledata.scverse.org/sbi2-cytodata/workshop/autoencoder_full.pt) | The same model trained on the full tier (gives `X_ae_fulltrained`) | 2 MB |
| [workshop_small_collection.zarr.zip](https://exampledata.scverse.org/sbi2-cytodata/workshop/workshop_small_collection.zarr.zip) | The small tier as a pre-shuffled annbatch collection (features and crops), for notebook 3 | 527 MB |
| [openphenom_demo_crops224.h5](https://exampledata.scverse.org/sbi2-cytodata/workshop/openphenom_demo_crops224.h5) | 256 cells at 224 px for the live OpenPhenom run in notebook 2 | 39 MB |

The hands-on parts need `workshop_small.h5ad`, `workshop_small_collection.zarr.zip`, `autoencoder.pt` and `openphenom_demo_crops224.h5` (about 1.1 GB).

## Schedule

| Block | Time | Notebook |
|---|---|---|
| Intro: scverse, AnnData, annbatch, deep learning on morphology data | 30 min | slides; `01_anndata.ipynb` as a warm-up |
| Inference with a pretrained image model (Tim) | 40 min | `02_inference_openphenom.ipynb` |
| Training from scratch with annbatch (Ilan) | 40 min | `03_training_annbatch.ipynb` |
| Questions and outlook | 10 min | |

## Contents

- `notebooks/01_anndata.ipynb`: the AnnData layout, one cell's image next to its features, per-well profiles with mantispy, neighbor graph and UMAP.
- `notebooks/02_inference_openphenom.ipynb`: the crops in `obsm`, preprocessing and channel order, OpenPhenom inference on the CPU, a featurized AnnData analysed with mantispy (which compounds are active, do compounds with the same mechanism look alike), the comparison with cp_measure features and a fine-tuned model, and why normalizing against a handful of control wells produces false calls.
- `notebooks/03_training_annbatch.ipynb` (Ilan's annbatch parts): an autoencoder on cp_measure features, class-balanced sampling with annbatch, a small CNN trained on crops streamed from a pre-shuffled collection (plain vs class-balanced), and the same code on the full tier.

## How the data were made

`scripts/` holds the pipeline that built the files on an HPC cluster with SLURM: subset selection, crop extraction, cp_measure features, normalization and feature selection with mantispy, OpenPhenom embeddings, the autoencoders, the annbatch collections and an out-of-core training check (`poc_annbatch_ooc.py`). The scripts read their locations from the `SBI2_WORK` and `SBI2_SOURCE` environment variables. `scripts/cluster_requirements.lock.txt` pins the build environment.

Nothing here is tied to a date; to reuse the materials, update the data links in `notebooks/data.py`.
