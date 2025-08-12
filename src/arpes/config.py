import os
from pathlib import Path
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from arpes.io import load_data, load_folder, export_dataset
from arpes.plotting.stack_plot import stack_dispersion_plot
from arpes.utilities.conversion import convert_to_kspace
from arpes.corrections import fix_fermi_edge
from arpes.analysis.general import rebin, normalize
from arpes.analysis.filters import gaussian_filter_arr
from arpes import xarray_extensions

from arpes.endstations.base import add_endstation
from arpes.endstations.plugins import (
    MAESTROMicroARPESEndstation,
    SPEEMEndstation,
    ALGMainChamber,
    HHGEndstation,
)

for endstation in [
    MAESTROMicroARPESEndstation,
    SPEEMEndstation,
    ALGMainChamber,
    HHGEndstation,
]:
    add_endstation(endstation)
xr.set_options(keep_attrs=True)
plt.rcParams["image.cmap"] = "magma"

cwd = Path(os.getcwd())
# We allow for one level of nesting in the directory structure
# Nominally, the only subdirectories should specify the measurement date
if cwd.name == "notebooks":
    root = cwd.parent
    data_root = root / "data"
    exports_root = data_root / "exports"
    results_root = root / "results"
else:
    if cwd.parent.name != "notebooks":
        from warnings import warn as _warn

        _warn(
            "The designated directory structure isn't being met. "
            "The provided paths may not be correct."
        )

    measurement_date = cwd.name
    root = cwd.parent.parent
    data_root = root / "data" / measurement_date
    exports_root = data_root / "exports"
    results_root = root / "results"
