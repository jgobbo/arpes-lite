"""Implements data loading for the Beamline 7 (MAESTRO) ARPES experiments at ALS.

Common code is provided by a base class reflecting DAQ similarities between micro- and nanoARPES
at MAESTRO. This is subclassed for the individual experiments to handle some subtle differences
in how nanoARPES handles its spatial coordiantes (they are hierarchical) and in the spectrometers.
"""
import numpy as np

import xarray as xr
from arpes.endstations import (
    FITSEndstation,
    HemisphericalEndstation,
    SynchrotronEndstation,
)

__all__ = (
    "MAESTROMicroARPESEndstation",
    #    "MAESTRONanoARPESEndstation"
)


class MAESTROARPESEndstationBase(
    SynchrotronEndstation, HemisphericalEndstation, FITSEndstation
):
    """Common code for the MAESTRO ARPES endstations at the Advanced Light Source."""

    PRINCIPAL_NAME = None  # skip me
    ALIASES = None  # skip me
    ANALYZER_INFORMATION = None

    # @override py3.12
    def postprocess_scan(self, data: xr.Dataset, scan_desc: dict = None):
        data.attrs.update(self.ANALYZER_INFORMATION)

        if "GRATING" in data.attrs:
            data.attrs["grating_lines_per_mm"] = {
                "G201b": 600,
            }.get(data.attrs["GRATING"])

        if "scan_x" in data.coords:
            # data.sel(scan_x=slice(None, None, -1)) #J: this might be a simpler way to do all below
            for data_var in data.data_vars:
                if "spectrum" in data_var:
                    data[data_var].values = np.flip(
                        data[data_var].values, axis=data[data_var].dims.index("scan_x")
                    )

        data = super().postprocess_scan(data, scan_desc)

        return data


class MAESTROMicroARPESEndstation(MAESTROARPESEndstationBase):
    """Implements data loading at the microARPES endstation of ALS's MAESTRO."""

    PRINCIPAL_NAME = "ALS-BL7"
    ALIASES = ["BL7", "BL7.0.2", "ALS-BL7.0.2", "MAESTRO"]

    ANALYZER_INFORMATION = {
        "analyzer": "R4000",
        "analyzer_name": "Scienta R4000",
        "parallel_deflectors": False,
        "perpendicular_deflectors": True,
        "analyzer_radius": None,
        "analyzer_type": "hemispherical",
    }

    RENAME_KEYS = {
        "LMOTOR0": "x",
        "LMOTOR1": "y",
        "LMOTOR2": "z",
        "Scan X": "scan_x",
        "Scan Y": "scan_y",
        "Scan Z": "scan_z",
        "LMOTOR3": "theta",
        "LMOTOR4": "beta",
        "LMOTOR5": "chi",
        "LMOTOR6": "alpha",
        "LMOTOR9": "psi",
        "mono_eV": "hv",
        "SF_HV": "hv",
        "SS_HV": "hv",
        "Slit Defl": "psi",
        "S_Volts": "volts",
        # probably need something like an attribute list for extraction
        "SFRGN0": "fixed_region_name",
        "SFE_0": "daq_center_energy",
        "SFLNM0": "lens_mode_name",
        "SFPE_0": "pass_energy",
        "UNDHARM": "undulator_harmonic",
        "RINGCURR": "beam_current",
        "SFFR_0": "frames_per_slice",
        "SFBA_0": "phi_prebinning",
        "SFBE0": "eV_prebinning",
        "LWLVNM": "daq_type",
        "pixel": "phi",
        "Swept_Spectra119": "spectrum",
        "Fixed_Spectra119": "spectrum",
    }

    ATTR_TRANSFORMS = {
        "START_T": lambda l: {
            "time": " ".join(l.split(" ")[1:]).lower(),
            "date": l.split(" ")[0],
        },
        "SF_SLITN": lambda l: {
            "slit_number": int(l.split(" ")[0]),
            "slit_shape": l.split(" ")[-1].lower(),
            "slit_width": float(l.split(" ")[2]),
        },
    }

    MERGE_ATTRS = {
        "mcp_voltage": None,
        "repetition_rate": 5e8,
        "undulator_type": "elliptically_polarized_undulator",
        "undulator_gap": None,
        "undulator_z": None,
        "undulator_polarization": None,
    }

    # @override py3.12
    def postprocess_scan(self, data: xr.Dataset, scan_desc: dict = None):
        data = super().postprocess_scan(data, scan_desc)

        conversion_factor = {
            "Angular30": 0.181,
            "Angular14": 0.0963,
            "Angular7NF": 0.04264,
        }[data.attrs["SSLNM0"]]

        if "phi" in data.coords:
            data.coords["phi"] = (conversion_factor / 200) * (
                data.coords["phi"] - data.coords["phi"].mean()
            )

        return data