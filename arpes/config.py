import os
from pathlib import Path
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from arpes.io import load_data, export_dataset
from arpes.plotting.stack_plot import stack_dispersion_plot
from arpes.utilities.conversion import convert_to_kspace
from arpes.corrections import fix_fermi_edge
from arpes.analysis.general import rebin, normalize
from arpes.analysis.filters import gaussian_filter_arr
from arpes import xarray_extensions

from arpes.endstations.base import add_endstation
from arpes.endstations.plugins.MAESTRO import MAESTROMicroARPESEndstation

add_endstation(MAESTROMicroARPESEndstation)

xr.set_options(keep_attrs=True)

cwd = Path(os.getcwd())
measurement_date = cwd.name
sample_name = cwd.parent.name
root = cwd.parent.parent.parent
data_root = root / "data" / sample_name / measurement_date
exports_root = data_root / "exports"
results_root = root / "results" / sample_name

plt.rcParams["image.cmap"] = "magma"
