from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr

from arpes.io import load_data, export_dataset
from arpes.plotting.qt_ktool import ktool
from arpes.utilities.conversion import convert_to_kspace
from arpes.corrections import fix_fermi_edge

from xarray import set_options

set_options(keep_attrs=True)

import os

cwd = Path(os.getcwd())
sample_name = cwd.name
data_root = cwd.parent.parent / "data" / sample_name
exports_root = data_root / "exports"
results_root = cwd.parent.parent / "results" / sample_name
