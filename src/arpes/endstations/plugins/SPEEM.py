"""Provides data loading for the Lanzara group SPEEM."""

import copy
from pathlib import Path

import pickle
import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt

import xarray as xr
from arpes.endstations import EndstationBase
from arpes.provenance import provenance_from_file
from arpes.constants import K_INV_ANGSTROM

__all__ = "SPEEMEndstation"

NS_PER_PIXEL = 4.576e-3
NOMINAL_TIMING_OFFSET = 921  # ns

# def pixels_to_time(
#     data: list,
#     pulse_delay: int = 1500,
#     time_per_pixel: float = 0.05,
#     nominal_delay: int = 1171,
# ):
#     return [pulse_delay - nominal_delay - value * time_per_pixel for value in data]

# def load_strict_range(
#     root: Path, x_range: tuple = None, y_range: tuple = None, t_range: tuple = None
# ):
#     with open(root / "raw_daq.pickle", "rb") as f:
#         raw_data = np.concatenate(pickle.load(f)["detector-frame-data"].data)

#     histogram: np.ndarray
#     bins = [
#         this_range[1] - this_range[0] if this_range is not None else 500
#         for this_range in [x_range, y_range, t_range]
#     ]
#     histogram, bins = np.histogramdd(
#         raw_data, bins=bins, range=[x_range, y_range, t_range]
#     )

#     # def midpoints(array: np.ndarray):
#     #     return [(a + b) / 2 for a, b in zip(array[:-1], array[1:])]
#     # coords = midpoints(bins)

#     coords = [bin[:-1] for bin in bins]

#     xr = DataArray(histogram, coords=coords, dims=["x", "y", "t"],)

#     return xr


class SPEEMEndstation(EndstationBase):
    """Provides data loading for the Lanzara group SPEEM."""

    PRINCIPAL_NAME = "ALG-SPEEM"
    ALIASES = ["SPEEM", "ToF", "PEEM-ToF"]
    _TOLERATED_EXTENSIONS = {".pickle"}

    @staticmethod
    def load_nc(filepath: Path, prune: bool = True):
        raw_data = xr.load_dataarray(filepath)
        if prune:
            raw_data = SPEEMEndstation.prune_counts(raw_data)

        return raw_data

    @staticmethod
    def prune_counts(count_list: np.ndarray) -> np.ndarray:
        t_min = float(count_list.min(axis=0)[2])
        return count_list[count_list[:, 2] != t_min]

    @staticmethod
    def coordinate_conversion(
        count_list: np.ndarray, conversion_table: np.ndarray, method: str = "cubic"
    ) -> np.ndarray:
        """
        Converts x, y, t detector coordinates to photoemission angle/position and
        energy.
        """
        radius = np.hypot(count_list[:, 0], count_list[:, 1])
        theta = np.arctan2(count_list[:, 1], count_list[:, 0])

        rt_count_list = np.column_stack((radius, count_list[:, 2]))
        r_data, t_data, ke_data, ang_data = conversion_table.transpose()

        radial_angle = griddata(
            np.column_stack((r_data, t_data)), ang_data, rt_count_list, method=method
        )
        kinetic_energy = griddata(
            np.column_stack((r_data, t_data)),
            ke_data,
            rt_count_list,
            method=method,
        )

        kp = K_INV_ANGSTROM * np.sqrt(kinetic_energy) * np.sin(np.deg2rad(radial_angle))
        kx = kp * np.cos(theta)
        ky = kp * np.sin(theta)

        converted_counts = np.column_stack((kx, ky, kinetic_energy))
        return converted_counts[np.where(~np.any(np.isnan(converted_counts), axis=1))]

    # def coordinate_conversion(
    #     self, count_list: np.ndarray, conversion_table: np.ndarray
    # ) -> np.ndarray:
    #     """
    #     Converts x, y, t detector coordinates to photoemission angle/position and
    #     energy.
    #     """
    #     radius = np.hypot(count_list[:, 0], count_list[:, 1])
    #     theta = np.arctan2(count_list[:, 1], count_list[:, 0])

    #     method = "cubic"
    #     kinetic_energy = griddata(
    #         conversion_table[:, 0:2],
    #         conversion_table[:, 2],
    #         count_list[:, 2],
    #         method=method,
    #     )
    #     radial_angle = griddata(
    #         conversion_table[:, 0:2], conversion_table[:, 3], radius, method=method
    #     )

    #     phi = np.multiply(radial_angle, np.cos(theta))
    #     psi = np.multiply(radial_angle, np.sin(theta))

    #     return np.stack((phi, psi, kinetic_energy), axis=1)

    def load(self, scan_desc: dict, conversion_table: np.ndarray = None, **kwargs):
        """
        Loads a pickle file from the SPEEM DAQ.

        Args:
            scan_desc: Dictionary with extra information to attach to the
            xarray.Dataset, must contain the location of the file

        Returns:
            The loaded spectrum.
        """

        metadata = copy.deepcopy(scan_desc)

        data_loc = metadata["file"]

        dataset_contents = dict()

        with open(data_loc, "rb") as f:
            raw_counts = pickle.load(f)["detector-frame-data"].data
        # concatenate = kwargs.get("concatenate", True)
        # raw_counts = [np.concatenate(raw_counts)] if concatenate else raw_counts
        # dataset_contents["raw_counts"] = raw_counts

        raw_counts: np.ndarray = np.concatenate(raw_counts)
        converted_counts = self.coordinate_conversion(raw_counts, conversion_table)

        n_bins = kwargs.get("bins", 100)
        histogram, bins = np.histogramdd(converted_counts, bins=n_bins)
        coords = {"phi": bins[0], "psi": bins[1], "KE": bins[2]}
        dataset_contents["raw"] = xr.DataArray(
            histogram, coords=coords, dims=tuple(coords.keys())
        )

        # timing_delay = (
        #     kwargs.get("timing_delay", NOMINAL_TIMING_OFFSET) - NOMINAL_TIMING_OFFSET
        # )  # ns
        # energy_offset = kwargs.get("energy_offset", 0)  # eV
        # energy_scale = kwargs.get("energy_scale", 16)  # ns/eV

        # datasets = []
        # for frame in raw_counts:
        #     pruned_frame = self.prune_data(frame)
        #     histogram, bins = np.histogramdd(pruned_frame, bins=n_bins)
        #     coords = {"phi": bins[0][1:], "psi": bins[1][1:], "KE": bins[2][1:]}

        #     dataset_contents["raw"] = xr.DataArray(
        #         histogram,
        #         coords=coords,
        #         # coords={
        #         #     "x": np.linspace(-12, 12, bins),
        #         #     "y": np.linspace(-12, 12, bins),
        #         #     "Eb": np.linspace(
        #         #         (timing_delay - 4095 * NS_PER_PIXEL) / energy_scale + energy_offset,
        #         #         timing_delay / energy_scale + energy_offset,
        #         #         bins,
        #         #     ),
        #         # },
        #         dims=tuple(coords.keys()),
        #         # attrs=f["/PRIMARY"].attrs.items(),
        #     )

        #     datasets.append(dataset_contents)

        provenance_from_file(
            dataset_contents["raw"],
            data_loc,
            {
                "what": "Loaded SPEEM dataset.",
                "by": "SPEEM.load",
            },
        )

        # if concatenate:
        #     return xr.Dataset(datasets[0], attrs=metadata)
        return xr.Dataset(dataset_contents, attrs=metadata)

    @staticmethod
    def plot_conversion_table_contours(conversion_table: np.ndarray):
        ke = conversion_table[:, 2]
        n_angs = int(np.argmax(np.diff(ke) != 0)) + 1
        n_kes = conversion_table.shape[0] // n_angs

        assert n_angs * n_kes == conversion_table.shape[0]

        ke_contours = [
            conversion_table[n_kes * i : n_kes * (i + 1), 0:2] for i in range(n_angs)
        ]
        ang_contours = [conversion_table[i::n_angs, 0:2] for i in range(n_kes)]

        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        for ke_contour in ke_contours:
            ax.plot(ke_contour[:, 0], ke_contour[:, 1], "k")
        for ang_contour in ang_contours:
            ax.plot(ang_contour[:, 0], ang_contour[:, 1], "k")
        ax.set_title("conversion table contours")
        ax.set_xlabel("r")
        ax.set_ylabel("tof")

        return fig, ax
