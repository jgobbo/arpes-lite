"""Implements support for the Lanzara HHG lab."""

import re
import os
from pathlib import Path

import numpy as np
import pandas as pd

import xarray as xr

from arpes.endstations import HemisphericalEndstation
from ..utilities.pxt import read_single_pxt

__all__ = ("HHGEndstation",)


def get_associated_kaindl_files(reference_path: Path):
    name_match = re.match(
        r"([\w+]*_?scan_[0-9][0-9][0-9]_)[0-9][0-9][0-9]\.pxt", reference_path.name
    )

    if name_match is None:
        return [reference_path]

    fragment = name_match.groups()[0]
    components = list(reference_path.parent.glob(f"{fragment}*.pxt"))
    components.sort()

    return components


def read_ai_file(path: Path) -> pd.DataFrame:
    """Read metadata from the Kaindl _AI.txt files.

    Kayla and Conrad discovered that Scienta does not record these files in a
    standardized format, but instead puts an arbitrarily long header at the top of the
    file and sometimes omits the column names.

    By manual inspection, we determined that despite this, the columns appear consistent
    across files recorded in these two formats. The columns are:

    ["Elapsed Time (s)", "Main Chamber", "Garage", "Integrated Photo AI",
     "Photo AI", "Photocurrent", "Heater Power", "Temperature A",
     "Temperature B"]

    depending on whether the header is there or not we need to skip a variable number of
    lines. The way we are detecting this is to look for the presence of the header and
    if it is in the file use it as the previous line before the start of the data.
    Ultimately we defer loading to pandas.

    Otherwise, if the header is absent we look for a tab as the first line of data.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    first_line_no = None
    for i, line in enumerate(lines):
        if "\t" in line:
            first_line_no = i
            break

    column_names = [
        "Elapsed Time (s)",
        "Main Chamber",
        "Garage",
        "Integrated Photo AI",
        "Photo AI",
        "Photocurrent",
        "Heater Power",
        "Temperature A",
        "Temperature B",
    ]

    return pd.read_csv(path, sep="\t", skiprows=first_line_no, names=column_names)


class HHGEndstation(HemisphericalEndstation):
    """The Kaindl Tr-ARPES high harmonic generation setup."""

    PRINCIPAL_NAME = "HHG"
    ALIASES = []

    _TOLERATED_EXTENSIONS = {
        ".pxt",
    }
    _SEARCH_PATTERNS = (
        r"[\-a-zA-Z0-9_\w+]+scan_[0]*{}_[0-9][0-9][0-9]",
        r"[\-a-zA-Z0-9_\w+]+scan_[0]*{}",
    )

    RENAME_KEYS = {
        "Delay Stage": "delay",
    }

    def resolve_frame_locations(self, scan_desc: dict = None):
        """Find .pxt files associated to a potentially multi-cut scan.

        This is very similar to what happens on BL4 at the ALS. You can look at the code
        for MERLIN to see more about how this works, or in
        `find_kaindl_files_associated`.
        """
        if scan_desc is None:
            raise ValueError(
                "Must pass dictionary as file scan_desc to all endstation loading code."
            )

        original_data_loc = scan_desc.get("path", scan_desc.get("file"))
        p = Path(original_data_loc)
        assert p.exists(), f"File {original_data_loc} does not exist."

        return get_associated_kaindl_files(p)

    def load_single_frame(
        self, frame_path: str = None, scan_desc: dict = None, **kwargs
    ):
        data = read_single_pxt(frame_path)
        return xr.Dataset({"spectrum": data}, attrs=data.attrs)

    def concatenate_frames(
        self,
        frames=list[xr.Dataset],
        scan_desc: dict = None,
    ):
        """Concenate frames from individual .pxt files on the Kaindl setup.

        The unique challenge here is to look for and parse the motor positions file (if
        they exist) and add this as a coordinate. As in Beamline 4 at the ALS, these
        Motor_Pos file gives the scan coordinate which we need to concatenate along.
        """
        if len(frames) < 2:
            return super().concatenate_frames(frames)

        # determine which axis to stitch them together along, and then do this
        original_filename = scan_desc.get("path", scan_desc.get("file"))

        internal_match = re.match(
            r"([a-zA-Z0-9\w+_]+)_[0-9][0-9][0-9]\.pxt", Path(original_filename).name
        )
        if internal_match.groups():
            motors_path = str(
                Path(original_filename).parent
                / f"{internal_match.groups()[0]}_Motor_Pos.txt"
            )
            with open(motors_path, "r") as f:
                motor_names = f.readline().strip().split("\t")
            frame_positions = np.loadtxt(motors_path, delimiter="\t", skiprows=1)

            for frame, frame_position in zip(frames, frame_positions):
                for motor_name, motor_position in zip(
                    motor_names, np.atleast_1d(frame_position)
                ):
                    frame.coords[motor_name] = motor_position

            def combine_datasets(ds_list, shape: list, dim_names: list):
                assert len(ds_list) == np.prod(shape)

                for dim_name, subshape in zip(dim_names, shape):
                    ds_list = [
                        xr.concat(
                            ds_list[i : i + subshape], dim_name, coords="different"
                        )
                        for i in range(0, len(ds_list), subshape)
                    ]
                return ds_list[0]

            shape = [
                len(np.unique(individual_motor_positions))
                for individual_motor_positions in frame_positions.T
            ]
            try:
                return combine_datasets(frames, shape, motor_names)
            except AssertionError:
                n_positions = np.prod(shape)
                while n_positions > len(frames):
                    shape[-1] -= 1
                    n_positions = np.prod(shape)
                return combine_datasets(frames[:n_positions], shape, motor_names)

    def postprocess_scan(self, data: xr.Dataset, scan_desc: dict = None):
        """Peform final data preprocessing for the HHG lab Tr-ARPES setup.

        This is very similar to what happens at BL4/MERLIN because the code was adopted
        from an old version of the DAQ on that beamline.
        """
        original_filename = Path(scan_desc.get("path", scan_desc.get("file")))
        internal_match = re.match(
            r"([a-zA-Z0-9\w+_]+_[0-9][0-9][0-9])\.pxt", original_filename.name
        )
        all_filenames = get_associated_kaindl_files(original_filename)
        all_filenames = [
            os.path.join(f.parent, f"{f.stem}_AI.txt") for f in all_filenames
        ]

        def load_attr_for_frame(filename, attr_name):
            # this is rereading which is not ideal but can be adjusted later
            df = read_ai_file(Path(filename))
            return df[attr_name][1:].to_numpy(dtype=float).mean()

        def attach_attr(data: xr.Dataset, attr_name, as_name):
            attributes = np.array(
                [load_attr_for_frame(f, attr_name) for f in all_filenames]
            )

            if len(attributes) == 1:
                data[as_name] = attributes[0]
            else:
                non_spectrometer_dims = [
                    d for d in data.data_vars["spectrum"].dims if d not in {"eV", "phi"}
                ]
                non_spectrometer_coords = {
                    c: v
                    for c, v in data.data_vars["spectrum"].coords.items()
                    if c in non_spectrometer_dims
                }

                new_shape = [len(data.coords[d]) for d in non_spectrometer_dims]
                n_positions = np.prod(new_shape)
                attributes_arr = xr.DataArray(
                    attributes[:n_positions].reshape(new_shape),
                    coords=non_spectrometer_coords,
                    dims=non_spectrometer_dims,
                )

                data = xr.merge([data, xr.Dataset(dict([[as_name, attributes_arr]]))])

            return data

        try:
            data = attach_attr(data, "Photocurrent", "photocurrent")
            data = attach_attr(data, "Temperature B", "temp")
            data = attach_attr(data, "Temperature A", "cryotip_temp")
        except FileNotFoundError as e:
            print(e)

        if internal_match.groups():
            attrs_path = str(
                Path(original_filename).parent / f"{internal_match.groups()[0]}_AI.txt"
            )

            try:
                extra = pd.read_csv(attrs_path, sep="\t", skiprows=6)
                data = data.assign_attrs(extra=extra.to_json())
            except Exception:
                # WELP we tried
                pass

        deg_to_rad_coords = {"theta", "beta", "phi"}

        for c in deg_to_rad_coords:
            if c in data.dims:
                data.coords[c] = data.coords[c] * np.pi / 180

        deg_to_rad_attrs = {"theta", "beta", "alpha", "chi"}
        for angle_attr in deg_to_rad_attrs:
            if angle_attr in data.attrs:
                data.attrs[angle_attr] = float(data.attrs[angle_attr]) * np.pi / 180

        ls = [data] + data.S.spectra
        for l in ls:
            l.coords["x"] = np.nan
            l.coords["y"] = np.nan
            l.coords["z"] = np.nan

        data = super().postprocess_scan(data, scan_desc)

        return data
