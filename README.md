# ARPES-Lite

ARPES-Lite is my personal fork of the original PyARPES package. I ran into some issues when first working with the package and fixing them required that I break a significant portion of the codebase.

The main improvements to pyarpes are in automation of loading and correcting data (fermi-edge correction, k-space conversion, etc.). There are also improvements and additions to QT based interactive tools.

Because of the changes made to data loading, some work is required to port endstations plugins from pyarpes. If needed, I can easily port existing endstations or add new ones.

## Installation

Arpes-lite is most easily installed with pip: `pip install arpes-lite`.

Note that the package is still imported with `import arpes` as with pyarpes.

For an editable installation, I recommend installing [uv](https://docs.astral.sh/uv/), cloning the github repo, navigating to the cloned repo in your shell, and installing with `uv sync`. This will also install ipykernel so you can run ipython notebooks with VS Code.

## Usage

The [PyARPES docs](https://arpes.readthedocs.io/en/latest/) are still the best place to learn how to use arpes-lite. I don't have my own docs to document all of the changes in usage, but there aren't many once you've gotten started.

To start, import common functions with `from arpes.config import *`. This contains data loading functions, imports the xarray_extensions, and other common functions for correcting and plotting data (fermi-edge correction, stack_plot, etc.) Having an editable `arpes.config` is the primary advantage of an editable installation.

Data loading can be done the same as in pyarpes with `load_data` or more easily with `load_folder`.

Here's an example of the code I use to load all of the data from an experiment:
```python
corrected: list[xr.Dataset] = [
    fix_fermi_edge(data)
    for data in load_folder(data_root, location="BL7", pattern="*.h5")
]
for i, data in enumerate(corrected):
    print(f"{i}: {data.attrs["comment"]}")
```
This will load all of the data in the `data_root` folder with a .h5 extension and then perform automatic fermi-edge correction on each dataset. Finally it prints a "comment" attribute stored during the measurement. From there you can `convert_to_kspace` without having to specify your momentum coordinates. Finally, you can use `export_dataset` to save a .nc file with any conversions/corrections you've performed so you don't have to do them every time you analyze your data.