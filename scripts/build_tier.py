"""Assemble one tier's AnnData (workshop_small.h5ad / workshop_full.h5ad) from selection, cp_measure, embeddings, crops.

X        cp_measure features, mantispy mad_robustize per plate over all cells (--norm plate, default) or vs DMSO per
         source (--norm source_dmso)
layers   raw: the un-normalized cp_measure features, selected features only (NaN elsewhere), float32
var      mantispy feature annotation (parsed from the cp_measure names), var["selected"] from pp.feature_select
obs      mantispy schema (Metadata_*) plus friendly columns (compound, target, is_control, ...) and qc_* flags
obsm     X_openphenom (off-the-shelf, <work>/X_openphenom.npy), X_openphenom_ft (run 05, joined by row_key),
         X_ae (<work>/X_ae.npy, when train_autoencoder.py has run), single_cell_images (n,5,96,96) uint8 and
         masks (n,2,96,96) uint8 as HDF5 datasets chunked (16,5,96,96) with --img-compression

Re-runnable: rerun after training the autoencoder / embedding to add X_ae / X_openphenom.
sbatch: scripts/run.sh 8 96G 120 scripts/build_tier.py --tier full
"""

import os
import argparse
import json
from pathlib import Path

import h5py
import zarr

import anndata as ad
import mantispy as mp
import numpy as np
import pandas as pd

W = Path(os.environ["SBI2_WORK"])
FT = Path(os.environ["SBI2_SOURCE"]) / Path("results/embeddings/05_JUMP_OpenPhenom_ECFP4_C_singlecell.h5ad")
OBJECTS = {"nucleus": "Nuclei", "cell": "Cells"}
# cp_measure names Haralick features without a group ("Variance_3_00_256__AGP"); mantispy's cp_measure grammar
# expects <Group>_<Feature>, so prefix them with Texture_ to get them parsed.
HARALICK = ("AngularSecondMoment", "Contrast", "Correlation", "Variance", "InverseDifferenceMoment", "SumAverage",
            "SumVariance", "SumEntropy", "Entropy", "DifferenceVariance", "DifferenceEntropy", "InfoMeas1", "InfoMeas2")


def cpm_name(c: str) -> str:
    head = c.split("_", 1)[0]
    is_texture = head in HARALICK and c.split("__")[0].count("_") == 3  # e.g. Contrast_3_00_256
    return f"Texture_{c}" if is_texture else c


ap = argparse.ArgumentParser()
ap.add_argument("--tier", choices=["small", "full"], default="small")
ap.add_argument("--norm", choices=["plate", "source_dmso"], default="plate")
ap.add_argument("--img-compression", choices=["gzip", "lzf"], default="gzip")  # gzip-4 beat lzf on size and read speed
args = ap.parse_args()
WORK = W / "work" if args.tier == "small" else W / "work/full"
SELECTED = W / ("data/selected.parquet" if args.tier == "small" else "data/full_selected.parquet")
OUT = W / f"workshop_{args.tier}.h5ad"

sel = pd.read_parquet(SELECTED)
parts = sorted((WORK / "cpm").glob("part_*.parquet"))
feat = pd.concat([pd.read_parquet(p) for p in parts]).sort_index()
assert len(feat) == len(sel) and (feat.row_key.to_numpy() == sel.row_key.to_numpy()).all(), "cp_measure rows misaligned"
feat = feat.drop(columns="row_key")
print(f"{len(parts)} parts, {feat.shape} features")

# ---- obs: mantispy schema + friendly columns
obs = pd.DataFrame(index=sel.row_key.to_numpy())
obs["Metadata_Source"] = sel.source.to_numpy()
obs["Metadata_Batch"] = sel.batch.to_numpy()
obs["Metadata_Plate"] = sel.plate.to_numpy()
obs["Metadata_Well"] = sel.well.to_numpy()
obs["Metadata_Site"] = sel.tile.astype(str).to_numpy()
obs["Metadata_ObjectNumber"] = sel.id_local.to_numpy()
obs["Metadata_Perturbation"] = sel.compound.to_numpy()
obs["Metadata_Perturbation_Type"] = "compound"
obs["Metadata_Compound"] = sel.compound.to_numpy()
obs["Metadata_InChIKey"] = sel.Metadata_InChIKey.to_numpy()
obs["Metadata_PlateType"] = sel.Metadata_PlateType.to_numpy()
obs["Metadata_LocalDensity"] = sel.n_cells_scaled.to_numpy()
obs["compound"] = sel.compound.to_numpy()
obs["target"] = sel.target.fillna("").to_numpy()
obs["is_control"] = sel.is_control.to_numpy()
# activity_class: re-checked on this tier's sources (recheck_curated.py); activity_class_selection: the class the
# compound was picked with (sources 4/13/5/11); "other" = not curated, "control" = DMSO
recheck = pd.read_parquet(W / "data/curated_recheck.parquet").set_index("Metadata_InChIKey").class_new
obs["activity_class_selection"] = sel.activity_class.to_numpy()
obs["activity_class"] = sel.Metadata_InChIKey.map(recheck).fillna(sel.activity_class).to_numpy()
obs["selection_map"] = sel.map_mean.to_numpy()  # fine-tuned run-05 activity mAP vs DMSO (mean over sources)
curated = set(pd.read_parquet(W / "data/selected.parquet").compound)  # the small tier's 40 compounds + DMSO
obs["highlight"] = np.where(sel.compound.isin(curated), sel.compound, "other")
obs["source"], obs["plate"], obs["well"] = sel.source.to_numpy(), sel.plate.to_numpy(), sel.well.to_numpy()
for c in ("nuc_area", "cyto_area", "nuc_cyto_ratio", "n_cells_in_fov", "row_key", "shard_index", "key_in_shard"):
    obs[c] = sel[c].to_numpy()

# ---- features and var via mantispy's cp_measure name parser, one object at a time
Xs, vars_ = [], []
for prefix, obj in OBJECTS.items():
    cols = [c for c in feat.columns if c.startswith(prefix + "_")]
    df = feat[cols].rename(columns=lambda c: cpm_name(c.split("_", 1)[1]))
    df = df.assign(Metadata_Plate=obs.Metadata_Plate.to_numpy(), Metadata_Well=obs.Metadata_Well.to_numpy())
    a = mp.io.from_dataframe(df.reset_index(drop=True), convention="cp_measure", resolution="object",
                             object_type="Cells")
    v = a.var.copy()
    v["object"] = obj
    v.index = [f"{obj}_{n}" for n in v.index]
    Xs.append(np.asarray(a.X, dtype=np.float32))
    vars_.append(v)
var = pd.concat(vars_)
X = np.hstack(Xs)
print("var feature_kind:", var.feature_kind.value_counts().to_dict(), "| dropped by parser:",
      feat.shape[1] - X.shape[1])

adata = ad.AnnData(X=X, obs=obs, var=var)
adata.obs = mp._core.schema.ensure_object_identity(adata.obs, object_type="Cells")
mp.pp.annotate_controls(adata, negcon=["DMSO"], perturbation_key="Metadata_Perturbation")
mp.io.stamp(adata, resolution="object", perturbation={"column": "Metadata_Perturbation",
                                                      "control_column": "Metadata_Control_Type"})

# ---- QC flags
nuc_cols, cell_cols = var.object.eq("Nuclei").to_numpy(), var.object.eq("Cells").to_numpy()
always_nan = np.isnan(X).mean(0) > 0.99
adata.obs["qc_empty_nucleus"] = np.isnan(X[:, nuc_cols & ~always_nan]).all(1)
adata.obs["qc_empty_cell"] = np.isnan(X[:, cell_cols & ~always_nan]).all(1)
nan_frac = np.isnan(X[:, ~always_nan]).mean(1)
adata.obs["qc_nan_fraction"] = nan_frac.astype(np.float32)
adata.obs["qc_high_nan"] = nan_frac > 0.1
area = adata.obs.nuc_area.to_numpy()
adata.obs["qc_tiny_nucleus"] = area < 50
adata.obs["qc_fail"] = (adata.obs.qc_empty_nucleus | adata.obs.qc_empty_cell | adata.obs.qc_high_nan
                        | adata.obs.qc_tiny_nucleus)
print("QC:", {c: int(adata.obs[c].sum()) for c in adata.obs.columns if c.startswith("qc_") and c != "qc_nan_fraction"})

# ---- normalize, then feature selection
raw = adata.X.copy()
if args.norm == "plate":
    mp.pp.normalize(adata, method="mad_robustize", by="Metadata_Plate", reference=None)
    norm = {"method": "mantispy.pp.normalize(method='mad_robustize', by='Metadata_Plate', reference=None)",
            "reference": "all cells of the same plate (in this tier)",
            "why": "TARGET2 has too few DMSO cells for DMSO-referenced statistics (one source in the full tier has a "
                   "single DMSO cell); in the sanity check whole-plate normalization halved the inactive compounds "
                   "called active (0.65 -> 0.30) relative to per-source DMSO normalization"}
else:
    mp.pp.normalize(adata, method="mad_robustize", by="Metadata_Source", reference="negcon")
    norm = {"method": "mantispy.pp.normalize(method='mad_robustize', by='Metadata_Source', reference='negcon')",
            "reference": "DMSO cells (Metadata_Control_Type == 'negcon') of the same source"}
mp.pp.feature_select(adata, key_added="selected")
selected = adata.var.selected.to_numpy()
print(f"selected {int(selected.sum())} / {adata.n_vars} features")
# size: raw only for the selected features (NaN elsewhere, compresses to ~nothing), float32
raw[:, ~selected] = np.nan
adata.layers["raw"] = raw
adata.uns["normalization"] = {
    **norm,
    "raw": "layers['raw']: un-normalized cp_measure features, selected features only (NaN elsewhere), float32",
    "feature_select": "mantispy.pp.feature_select defaults; var['selected']",
    "clipping": "none in X; notebooks clip at |z| <= 50",
    "cp_measure": "cp_measure 0.3.0 featurizer, objects nucleus+cell (mask>0), 5 channels scaled to [0,1], "
                  "Costes scale_max=255, 224 px crops",
}

# ---- embeddings
op = WORK / "X_openphenom.npy"
if op.exists():
    adata.obsm["X_openphenom"] = np.load(op)
ft = ad.read_h5ad(FT, backed="r")
keys = ft.obs["row_key"].to_numpy() if "row_key" in ft.obs else ft.obs_names.to_numpy()
pos = pd.Series(np.arange(len(keys)), index=keys).reindex(adata.obs.row_key.to_numpy())
assert pos.notna().all(), f"fine-tuned join coverage {pos.notna().mean():.4f} < 100%"
idx = pos.astype(int).to_numpy()
order = np.argsort(idx)
emb = np.empty((len(idx), ft.shape[1]), np.float32)
emb[order] = np.asarray(ft.X[np.sort(idx)], dtype=np.float32)
adata.obsm["X_openphenom_ft"] = emb
print("fine-tuned join: 100% coverage,", emb.shape)
ae = WORK / "X_ae.npy"
if ae.exists():
    adata.obsm["X_ae"] = np.load(ae)
adata.uns["obsm_description"] = json.dumps({
    "X_openphenom": "off-the-shelf recursionpharma/OpenPhenom, 384-d, 224->256 px bilinear, uint8",
    "X_openphenom_ft": "CP-BG-Bench run 05 (OpenPhenom + LoRA, ECFP4 CLIP, view C) 128-d, joined by row_key",
    "X_ae": "latent of the pretrained MLP autoencoder (autoencoder.pt, trained on the small tier) on X[:, features "
            "stored in the checkpoint], clipped |z| <= 50, NaN -> 0",
    "single_cell_images": "(n, 5, 96, 96) uint8 [DNA, AGP, ER, Mito, RNA], 224 px crops downsampled with cv2 INTER_AREA",
    "masks": "(n, 2, 96, 96) uint8 {0, 255} [nucleus, cell], downsampled with cv2 INTER_NEAREST",
})
adata.uns["image_channels"] = ["DNA", "AGP", "ER", "Mito", "RNA"]
adata.uns["mask_channels"] = ["nucleus", "cell"]
adata.uns["tier"] = args.tier
for c in adata.obs.columns:
    if adata.obs[c].dtype == object or pd.api.types.is_string_dtype(adata.obs[c]):
        adata.obs[c] = adata.obs[c].astype("category")
adata.write_h5ad(OUT, compression="gzip")

# ---- crops: 4D obsm arrays as chunked, compressed HDF5 datasets
crops = zarr.open_group(WORK / "crops96.zarr", mode="r")
assert (ad.io.read_elem(crops["obs"]).row_key.to_numpy() == adata.obs.row_key.to_numpy()).all(), "crops misaligned"
comp = {"compression": "gzip", "compression_opts": 4} if args.img_compression == "gzip" else {"compression": "lzf"}
with h5py.File(OUT, "a") as f:
    for name, src in (("single_cell_images", "images"), ("masks", "masks")):
        arr = crops[src][:]
        ad.io.write_elem(f["obsm"], name, arr, dataset_kwargs={"chunks": (16, *arr.shape[1:]), **comp})
print(ad.read_h5ad(OUT, backed="r"))
