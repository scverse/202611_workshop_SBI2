# Notebooks

Run them in order; each one downloads what it needs into `data/` (here, `notebooks/data/`) and switches between the small and the full tier with `TIER = "small"` or `"full"`.

| Notebook | What it does | Runtime / peak RAM (small tier, 4 CPU threads, measured on a cluster node) |
|---|---|---|
| `01_anndata.ipynb` | Warm-up: AnnData layout, one cell's image and features, per-well profiles with mantispy, neighbor graph and UMAP | about 1 min / 1.0 GB |
| `02_inference_openphenom.ipynb` | The crops in `obsm`, preprocessing and channel order, OpenPhenom inference live (`RUN_LIVE`, `N_CELLS`), a featurized AnnData, mantispy activity and mechanism consistency, cp_measure vs fine-tuned embeddings, and what normalizing against few controls does | about 1.5 min / 1.6 GB with `RUN_LIVE = False` (live inference of 1,000 cells adds about 2-3 min) |
| `03_training_annbatch.ipynb` | Ilan's annbatch parts: feature autoencoder, class-balanced sampling, a small CNN on crops streamed from a collection (plain vs balanced), the full tier | about 3.5 min / 3.7 GB (two CNN runs of 3 epochs each, about 1 min each) |

`data.py` holds the download registry (URLs and sha256) and small helpers for loading and plotting. `src/` holds the notebooks as plain Python scripts (jupytext percent format), from which the `.ipynb` files are generated and executed.
