"""Utility functions for extracting ARPES information from the HDF5 file conventions."""

import numpy as np
import h5py
from .common import safe_reshape

__all__ = (
    "construct_coords",
    "get_attrs",
    "dataset_to_array",
)


def get_scan_coords(
    scan_info: dict[str, str], scalar_data: h5py.Dataset
) -> dict[str, np.ndarray]:
    """Gets the scan coordinates from the scan information in the headers"""

    n_loops = int(scan_info["LWLVLPN"])
    scan_coords = {}
    for loop in range(n_loops):
        n_scan_coords = int(scan_info[f"NMSBDV{loop}"])
        n_scan_dimensions = 0
        for i in range(n_scan_coords):
            if f"ST_{loop}_{i}" in scan_info:
                n_scan_dimensions += 1

        shape = tuple(
            [int(scan_info[f"N_{loop}_{i}"]) for i in range(n_scan_dimensions)][::-1]
        )

        for scan_dimension in range(n_scan_dimensions):
            if (f"ST_{loop}_{scan_dimension}" not in scan_info) or (
                shape[scan_dimension] == 1
            ):
                continue
            name = scan_info[f"NM_{loop}_{scan_dimension}"]
            raw_data = dataset_to_array(scalar_data[name])
            reshaped_data = np.moveaxis(
                safe_reshape(raw_data, shape), scan_dimension, 0
            )
            averaged_data = reshaped_data.mean(axis=tuple(range(n_scan_dimensions - 1)))
            scan_coords[name] = averaged_data

    return scan_coords


def construct_coords(
    hdf5: h5py.File,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str]], tuple[str]]:
    """
    Constructs all coordinates from the HDF5 file, including the scan coordinates and the detector coordinates.
    Returns a dictionary of the coordinates and a dictionary of the dimensions for each data variable
    """

    scan_header = hdf5["Headers"]["Scan"] if "Scan" in hdf5["Headers"] else []
    low_level_scan_header = hdf5["Headers"]["Low_Level_Scan"]
    scan_info = {}
    for item in list(scan_header) + list(low_level_scan_header):
        item: list[bytes]
        try:
            scan_info[item[1].decode("utf-8").strip()] = (
                item[2].decode("utf-8").replace("'", "")
            )
        except UnicodeDecodeError:
            pass

    scan_coords = get_scan_coords(scan_info, hdf5["0D_Data"])
    scan_coord_names = tuple(scan_coords.keys())
    data_dimensions = {}
    constructed_coords = scan_coords.copy()

    for scalar_data in hdf5["0D_Data"]:
        if scalar_data in scan_coords:
            continue
        data_dimensions[scalar_data] = tuple(scan_coords.keys())

    for n_dims in (1, 2):
        dataset = hdf5[f"{n_dims}D_Data"]
        for data_name in dataset:
            attrs = dataset[data_name].attrs
            offsets = attrs["scaleOffset"][::-1]
            deltas = attrs["scaleDelta"][::-1]
            coord_names = attrs["unitNames"][::-1]
            coord_lengths = dataset[data_name].shape[:n_dims]

            all_coords_for_data = []
            for dim in range(n_dims):
                coord_name = coord_names[dim]
                delta = (
                    round(deltas[dim]) if coord_names[dim] == "pixel" else deltas[dim]
                )
                coord_values = np.linspace(
                    offsets[dim],
                    offsets[dim] + delta * coord_lengths[dim],
                    coord_lengths[dim],
                    endpoint=False,
                )

                if coord_name in constructed_coords:
                    if np.array_equal(coord_values, constructed_coords[coord_name]):
                        all_coords_for_data.append(coord_name)
                        continue
                    else:
                        coord_name = f"{coord_name}_{data_name}"

                all_coords_for_data.append(coord_name)
                constructed_coords[coord_name] = coord_values
            all_coords_for_data += list(scan_coords.keys())
            data_dimensions[data_name] = tuple(all_coords_for_data)
    return constructed_coords, data_dimensions, scan_coord_names


def get_attrs(hdf5: h5py.File) -> dict[str, str]:
    """Gets the relevant attributes from the HDF5 file"""

    attrs = {}
    try:
        comments = hdf5["Comments"]["PreScan"]
        attrs["comment"] = comments[0][0].decode("ascii")
    except KeyError:
        pass

    def clean_attr(value: bytes) -> float | str:
        value = value.decode("ascii")
        try:
            return float(value)
        except ValueError:
            return value.strip("'")

    column_for_name = {"Beamline": 3}
    skip_headers = ["Low_Level_Scan", "Scan", "Switch"]
    headers = hdf5["Headers"]
    for header in headers:
        if header in skip_headers:
            continue

        header_attrs = {
            item[column_for_name.get(header, 0)].decode("ascii"): clean_attr(item[2])
            for item in headers[header]
        }
        attrs.update(header_attrs)

    return attrs


def dataset_to_array(dataset: h5py.Dataset, type: str = "float64") -> np.ndarray:
    """Quickly converts a h5py dataset to a numpy array"""
    # Converting to a numpy array without using .astype takes ~1000x longer
    return dataset.astype(type)[:]
