from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr

from arpes.io import load_data, export_dataset
from arpes.plotting.qt_ktool import ktool
from arpes.plotting.stack_plot import stack_dispersion_plot
from arpes.utilities.conversion import convert_to_kspace
from arpes.corrections import fix_fermi_edge

from xarray import set_options

set_options(keep_attrs=True)

import os

cwd = Path(os.getcwd())
measurement_date = cwd.name
sample_name = cwd.parent.name
root = cwd.parent.parent.parent
data_root = root / "data" / sample_name / measurement_date
exports_root = data_root / "exports"
results_root = root / "results" / sample_name

plt.rcParams["image.cmap"] = "magma"
