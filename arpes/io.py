"""Provides the core IO facilities supported by PyARPES.

The most important here are the data loading functions (load_data, load_example_data).
and pickling utilities.

Heavy lifting is actually performed by the plugin definitions which know how to ingest
different data formats into the PyARPES data model.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from arpes.endstations import load_scan
from arpes.typing import DataType

__all__ = (
    "load_data",
    "load_folder",
    "export_dataset",
    "load_example_data",
    "list_pickles",
    "stitch",
)


def load_data(file: str | Path | int, location: str | type, **kwargs) -> xr.Dataset:
    """Load a piece of data using available plugins.

    Args
    ----
    file : An identifier for the file which should be loaded. If this is a number or
        can be coerced to one, data will be loaded from the workspace data folder if
        a matching unique file can be found for the number. If the value is a
        relative path, locations relative to the cwd and the workspace data folder
        will be checked. Absolute paths can also be used in a pinch.
    location : The name of the endstation/plugin to use. You should try to provide
        one. If None is provided, the loader will try to find an appropriate one
        based on the file extension and brute force. This will be slower and can be
        error prone in certain circumstances.

        Optionally, you can pass a loading plugin (the class) through this kwarg and
        directly specify the class to be used.


    Returns
    -------
        The loaded data. Ideally, data which is loaded through the plugin system should
        be highly compliant with the PyARPES data model and should work seamlessly with
        PyARPES analysis code.
    """
    try:
        file = int(str(file))
    except ValueError:
        file = str(Path(file).absolute())

    desc = {
        "file": file,
        "location": location,
    }

    return load_scan(desc, **kwargs)


def load_folder(
    folder: Path, location: str | type, pattern: str = "*", **kwargs
) -> list[xr.Dataset]:
    """Load all files in a folder.

    Args
    ----
    folder : Path
        The folder to load all data from
    location : str | type
        The endstation plugin to use for loading the data
    pattern : str, (default '*')
        The pattern to use for globbing the files in the folder. Defaults to all files

    Returns
    -------
        A list of all the datasets loaded from the specified folder
    """

    all_datasets = []
    for file in folder.glob(pattern):
        all_datasets.append(load_data(file, location, **kwargs))
    return all_datasets


def load_laue(path: Path | str):
    """Load NorthStart Laue backscattering data."""
    northstar_62_69_dtype = np.dtype(
        [
            ("pad1", "B", (2364,)),  # unused
            ("sample", "S52"),
            ("user", "S52"),
            ("comment", "S512"),
            ("pad2", "B", (228,)),  # unused
        ]
    )
    """
    Primitive support for loading Laue data from the NorthStar x-ray backscattering DAQ program.

    Laue file structure courtesy Jonathan Denlinger, MERLIN endstations at the ALS
    16-bit binary Laue histogram (.hs2) file
    Format:  2 byte*256*256= 131072 long + header info at the end
    Northstar 6.0:  132028 bytes   (956 byte extra)
    Northstar 6.2.6.9:  134280 bytes  ( 3208 byte extra)
    header includes (offset from start of header):
    byte 65536 / 131072 / 2364   = sample name (character string - read to double space)
    byte 65587 / 131124 / 2416  = operator (character string - read to double space)
    byte 65638 / 131176 / 2468  = date in mm/dd/yy format (8 character string)
    byte 65811 / 131760 / 2984  = dwell time * 10 in seconds (word)
    byte 65821 / 131776 / 3000  = mA
    byte 65823 / 131780 / 3004  = kV
    byte 131664 / 592 = index file name
    """
    if isinstance(path, str):
        path = Path(path)

    binary_data = path.read_bytes()
    table, header = binary_data[:131072], binary_data[131072:]

    table = np.fromstring(table, dtype=np.uint16).reshape(256, 256)
    header = np.fromstring(header, dtype=northstar_62_69_dtype).item()

    arr = xr.DataArray(
        table,
        coords={"x": np.array(range(256)), "y": np.array(range(256))},
        dims=[
            "x",
            "y",
        ],
        attrs={
            "sample": header[1].split(b"\0")[0].decode("ascii"),
            "user": header[2].split(b"\0")[0].decode("ascii"),
            "comment": header[3].split(b"\0")[0].decode("ascii"),
        },
    )
    return arr


DATA_EXAMPLES = {
    "cut": ("ALG-MC", "cut.fits"),
    "map": ("example_data", "fermi_surface.nc"),
    "photon_energy": ("example_data", "photon_energy.nc"),
    "nano_xps": ("example_data", "nano_xps.nc"),
    "temperature_dependence": ("example_data", "temperature_dependence.nc"),
}


def load_example_data(example_name="cut") -> xr.Dataset:
    """Provide sample data for executable documentation."""
    if example_name not in DATA_EXAMPLES:
        warnings.warn(
            f"Could not find requested example_name: {example_name}. Please provide one of {list(DATA_EXAMPLES.keys())}"
        )

    location, example = DATA_EXAMPLES[example_name]
    file = Path(__file__).parent / "example_data" / example
    return load_data(file=file, location=location)


@dataclass
class ExampleData:
    @property
    def cut(self) -> xr.Dataset:
        return load_example_data("cut")

    @property
    def map(self) -> xr.Dataset:
        return load_example_data("map")

    @property
    def photon_energy(self) -> xr.Dataset:
        return load_example_data("photon_energy")

    @property
    def nano_xps(self) -> xr.Dataset:
        return load_example_data("nano_xps")

    @property
    def temperature_dependence(self) -> xr.Dataset:
        return load_example_data("temperature_dependence")


example_data = ExampleData()


def stitch(
    df_or_list: list[str] | pd.DataFrame,
    attr_or_axis: str,
    built_axis_name: str = None,
    sort: bool = True,
) -> DataType:
    """Stitches together a sequence of scans or a DataFrame.

    Args:
        df_or_list: The list of the files to load
        attr_or_axis: Coordinate or attribute in order to promote to an index. I.e. if 't_a' is specified,
                      we will create a new axis corresponding to the temperature and concatenate the data along this axis
        built_axis_name: The name of the concatenated output dimensions
        sort: Whether to sort inputs to the concatenation according to their `attr_or_axis` value.

    Returns:
        The concatenated data.
    """
    list_of_files = None
    if isinstance(df_or_list, (pd.DataFrame,)):
        list_of_files = list(df_or_list.index)
    else:
        if not isinstance(df_or_list, (list, tuple)):
            raise TypeError(
                "Expected an interable for a list of the scans to stitch together"
            )

        list_of_files = list(df_or_list)

    built_axis_name = attr_or_axis if built_axis_name is None else built_axis_name

    if not list_of_files:
        raise ValueError("Must supply at least one file to stitch")

    loaded = [
        f if isinstance(f, (xr.DataArray, xr.Dataset)) else load_data(f)
        for f in list_of_files
    ]

    for i, loaded_file in enumerate(loaded):
        value = None
        if isinstance(attr_or_axis, (list, tuple)):
            value = attr_or_axis[i]
        elif attr_or_axis in loaded_file.attrs:
            value = loaded_file.attrs[attr_or_axis]
        elif attr_or_axis in loaded_file.coords:
            value = loaded_file.coords[attr_or_axis]

        loaded_file = loaded_file.assign_coords(dict([[built_axis_name, value]]))

    if sort:
        loaded.sort(key=lambda x: x.coords[built_axis_name])

    concatenated = xr.concat(loaded, dim=built_axis_name)
    if "id" in concatenated.attrs:
        del concatenated.attrs["id"]

    from arpes.provenance import provenance_multiple_parents

    provenance_multiple_parents(
        concatenated,
        loaded,
        {
            "what": "Stitched together separate datasets",
            "by": "stitch",
            "dim": built_axis_name,
        },
    )

    return concatenated


def export_dataset(dataset: xr.Dataset, path: str | Path) -> None:
    """Correct bad keys/values then export dataset to netcdf.

    Note that all values that are not strings, ints, or floats are converted to strings.

    Args
    ----
        dataset : xr.Dataset
            The dataset to export
        path : str | Path
            export path
    """
    path = Path(path) if isinstance(path, str) else path
    dataset = dataset.copy(deep=True)

    fixed_values = {}
    bad_keys = []
    for key, value in dataset.attrs.items():
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            fixed_values[key] = str(value)
        if "/" in key:
            bad_keys.append(key)
    for key in fixed_values:
        dataset.attrs[key] = fixed_values[key]
    for key in bad_keys:
        dataset.attrs[key.replace("/", " per ")] = dataset.attrs.pop(key)

    if path.suffix != ".nc":
        warnings.warn(
            "The provided path doesn't have a .nc extension. Adding one and continuing."
        )
        path = path.parent / f"{path.name}.nc"
    if not path.parent.exists():
        path.parent.mkdir()
    dataset.to_netcdf(path)


# TODO: J: add an import dataset function which converts bools/Nones back to their original types
