"""Write uns["autoencoder"] notes into both tiers (in place, h5py; crops untouched)."""
import os
from pathlib import Path

import anndata as ad
import h5py

W = Path(os.environ["SBI2_WORK"])
NOTES = {
    "small": {"X_ae": "latent of autoencoder.pt, trained on the small tier (the model participants train)"},
    "full": {"X_ae": "latent of autoencoder.pt, trained on the small tier (the model participants train)",
             "X_ae_fulltrained": "latent of autoencoder_full.pt, same architecture retrained on the full tier "
                                 "(val MSE 0.667 vs 0.800 for autoencoder.pt on the full validation split)"},
}
for tier, notes in NOTES.items():
    with h5py.File(W / f"workshop_{tier}.h5ad", "a") as f:
        if "autoencoder" in f["uns"]:
            del f["uns"]["autoencoder"]
        ad.io.write_elem(f["uns"], "autoencoder", notes)
        print(tier, list(f["obsm"].keys()))
