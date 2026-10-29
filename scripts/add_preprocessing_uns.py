"""Record the image preprocessing in uns["preprocessing"] of both tiers (in place; crops untouched)."""
import os
from pathlib import Path

import anndata as ad
import h5py

W = Path(os.environ["SBI2_WORK"])
PRE = {
    "source": "JUMP Cell Painting cpg0016, TARGET2 plates; single-cell crops from CP-BG-Bench (view C: raw crop)",
    "segmentation": "nucleus and cell masks from CP-BG-Bench's segmentation, stored as {0, 255}",
    "cell_sampling": "CP-BG-Bench: up to 50 cells per well (uniform_per_well, seed 42)",
    "cell_qc": "CP-BG-Bench: kept cells within the 2.5-97.5% quantiles of nucleus area, cytoplasm area and their "
               "ratio; this workshop adds obs['qc_fail'] (True for cells with an empty nucleus or cell mask)",
    "intensity": "per field of view and channel, clipped to the 0.1 and 99.9 percentiles and rescaled to uint8 "
                 "(CP-BG-Bench per_fov_percentile)",
    "crop": "150 px patch centred on the cell, resized to 224 px (CP-BG-Bench); stored here downsampled to 96 px "
            "(cv2 INTER_AREA for images, INTER_NEAREST for masks)",
    "channel_order": "obsm['single_cell_images'] channels follow uns['image_channels'] = [DNA, AGP, ER, Mito, RNA]; "
                     "JUMP's own order is [DNA, ER, RNA, AGP, Mito], Recursion's [DNA, ER, AGP, RNA, Mito, ...]",
    "openphenom_input": "X_openphenom: 224 px crops resized to 256 px (bilinear, antialias), uint8, channels in "
                        "Recursion order; OpenPhenom scales by 1/255 and instance-normalizes each channel itself",
    "cp_measure_input": "224 px crops scaled to [0, 1], masks > 0 as one object each (nucleus, cell)",
}
for tier in ("small", "full"):
    with h5py.File(W / f"workshop_{tier}.h5ad", "a") as f:
        if "preprocessing" in f["uns"]:
            del f["uns"]["preprocessing"]
        ad.io.write_elem(f["uns"], "preprocessing", PRE)
    print(tier, "uns['preprocessing'] written")
