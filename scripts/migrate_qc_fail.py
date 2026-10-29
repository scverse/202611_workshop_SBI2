"""mantispy 0.1.0a2 convention: replace obs["qc_pass"] with obs["qc_fail"] (= ~qc_pass) in place.

Rewrites only obs (h5ad: the obs group; collections: each dataset group's obs) and re-zips the collections.
Then checks that mantispy.pp.filter_cells(require_qc=True) works on each h5ad.
"""
import os
import subprocess
from pathlib import Path

import anndata as ad
import h5py
import mantispy as mt
import zarr

W = Path(os.environ["SBI2_WORK"])


def migrate(obs):
    if "qc_pass" in obs:
        obs.insert(obs.columns.get_loc("qc_pass"), "qc_fail", ~obs.pop("qc_pass").astype(bool))
    assert "qc_fail" in obs and "qc_pass" not in obs
    return obs


for tier in ("small", "full"):
    p = W / f"workshop_{tier}.h5ad"
    with h5py.File(p, "a") as f:
        obs = migrate(ad.io.read_elem(f["obs"]))
        del f["obs"]
        ad.io.write_elem(f, "obs", obs)
    with h5py.File(p, "r") as f:
        a = ad.AnnData(obs=ad.io.read_elem(f["obs"]), uns={"mantispy": ad.io.read_elem(f["uns"]["mantispy"])})
    n, n_fail = a.n_obs, int(a.obs["qc_fail"].sum())
    mt.pp.filter_cells(a, min_cells_per_well=0, require_qc=True)
    assert a.n_obs == n - n_fail
    print(f"{p.name}: {n_fail} of {n} cells qc_fail; mt.pp.filter_cells(require_qc=True) keeps {a.n_obs}")

for tier in ("small", "full"):
    c = W / f"workshop_{tier}_collection.zarr"
    for name, g in zarr.open_group(c, mode="a").groups():
        obs = migrate(ad.io.read_elem(g["obs"]))
        del g["obs"]
        ad.io.write_elem(g, "obs", obs)
        print(f"{c.name}/{name}: obs migrated ({int(obs['qc_fail'].sum())} qc_fail)")
    z = W / f"{c.name}.zip"
    z.unlink(missing_ok=True)
    subprocess.run(["zip", "-q", "-r", "-0", z.name, c.name], cwd=W, check=True)
    print(f"re-zipped {z.name}: {z.stat().st_size} bytes")
