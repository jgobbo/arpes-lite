%load_ext autoreload
%autoreload 2

import xarray as xr
import xrft
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.ndimage as ndi
import os.path

import bokeh
import holoviews as hv
import holoviews.util

from arpes.config import CONFIG, SOURCE_PATH, FIGURE_PATH

import arpes
from arpes.analysis import *
from arpes.io import *
from arpes.pipeline import compose
from arpes.pipelines import *
from arpes.plotting import *
from arpes.preparation import *

import arpes.xarray_extensions # Ensure that extensions get loaded

from arpes.models import load_scan
from arpes.utilities import clean_xlsx_dataset, default_dataset, swap_reference_map
from arpes.utilities.conversion import *

from bokeh.io import output_notebook

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]