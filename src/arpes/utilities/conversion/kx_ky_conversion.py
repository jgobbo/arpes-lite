"""Implements 2D and 3D angle scan momentum conversion for Fermi surfaces.

Broadly, this covers cases where we are not performing photon energy scans.
"""

import numpy as np

import numba
import math

import arpes.constants
import xarray as xr
from typing import Any, Callable

from .base import CoordinateConverter, K_SPACE_BORDER, MOMENTUM_BREAKPOINTS
from .bounds_calculations import calculate_kp_bounds, calculate_kx_ky_bounds

__all__ = ["ConvertKp", "ConvertKxKy"]


@numba.njit(parallel=True)
def _exact_arcsin(k_par, k_perp, k_tot, phi, offset, par_tot, negate):
    """A efficient arcsin with total momentum scaling."""
    mul_idx = 1 if par_tot else 0
    for i in numba.prange(len(k_par)):
        result = np.arcsin(k_par[i] / np.sqrt(k_tot[i * mul_idx] ** 2 - k_perp[i] ** 2))
        if negate:
            result = -result

        phi[i] = result + offset


@numba.njit(parallel=True)
def _small_angle_arcsin(k_par, k_tot, phi, offset, par_tot, negate):
    """A efficient small angle arcsin with total momentum scaling.

    np.arcsin(k_par / k_tot, phi)
    phi += offset
    mul_idx = 0
    """
    mul_idx = 1 if par_tot else 0
    for i in numba.prange(len(k_par)):
        result = np.arcsin(k_par[i] / k_tot[i * mul_idx])

        if negate:
            result = -result

        phi[i] = result + offset


@numba.njit(parallel=True)
def _rotate_kx_ky(kx, ky, kxout, kyout, chi):
    cos_chi = np.cos(chi)
    sin_chi = np.sin(chi)
    for i in numba.prange(len(kx)):
        kxout[i] = kx[i] * cos_chi - ky[i] * sin_chi
        kyout[i] = ky[i] * cos_chi + kx[i] * sin_chi


@numba.njit(parallel=True)
def _compute_ktot(hv, work_function, binding_energy, k_tot):
    for i in numba.prange(len(binding_energy)):
        k_tot[i] = arpes.constants.K_INV_ANGSTROM * math.sqrt(
            hv - work_function + binding_energy[i]
        )


def _safe_compute_k_tot(hv, work_function, binding_energy):
    arr_binding_energy = binding_energy
    if not isinstance(binding_energy, np.ndarray):
        arr_binding_energy = np.array([binding_energy])

    k_tot = np.zeros_like(arr_binding_energy)
    _compute_ktot(hv, work_function, arr_binding_energy, k_tot)

    return k_tot


class ConvertKp(CoordinateConverter):
    """A momentum converter for single ARPES (kp) cuts."""

    k_tot: np.ndarray = None
    phi: np.ndarray = None

    def __init__(self, ds: xr.Dataset, *args: Any, **kwargs: Any) -> None:
        """Initialize the kp momentum converter and cached coordinate values."""
        super().__init__(ds, *args, **kwargs)

        if self.is_slit_vertical:
            self.parallel_angle = "beta"
            self.perpendicular_angles = ["psi", "theta"]
        else:
            self.parallel_angle = "theta"
            self.perpendicular_angles = ["psi", "beta"]

    @property
    def phi_offset(self) -> float:
        """The offset for the phi angle."""
        return self.ds.S.phi_offset + self.ds.S.lookup_offset_coord(self.parallel_angle)

    @property
    def perpendicular_offset(self) -> float:
        """The offset for the perpendicular angle."""
        return sum(
            self.ds.S.lookup_offset_coord(angle) for angle in self.perpendicular_angles
        )

    def get_coordinates(
        self, resolution: dict = None, bounds: dict = None
    ) -> dict[str, np.ndarray]:
        """Calculates appropriate coordinate bounds."""
        resolution = {} if resolution is None else resolution
        bounds = {} if bounds is None else bounds

        coordinates = super().get_coordinates(resolution, bounds=bounds)
        if "kp" in bounds:
            kp_low, kp_high = bounds["kp"]
        else:
            kp_low, kp_high = calculate_kp_bounds(
                self.ds, self.phi_offset, self.perpendicular_offset
            )

        inferred_kp_res = (kp_high - kp_low + 2 * K_SPACE_BORDER) / len(
            self.ds.coords["phi"]
        )

        try:
            inferred_kp_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kp_res][
                -2 if (len(self.ds.coords["phi"]) < 80) else -1
            ]
        except IndexError:
            inferred_kp_res = MOMENTUM_BREAKPOINTS[-2]

        coordinates["kp"] = np.arange(
            kp_low - K_SPACE_BORDER,
            kp_high + K_SPACE_BORDER,
            resolution.get("kp", inferred_kp_res),
        )

        base_coords = {
            k: v
            for k, v in self.ds.coords.items()
            if k not in ["eV", "phi", "beta", "theta"]
        }

        coordinates.update(base_coords)
        return coordinates

    def compute_k_tot(self, binding_energy: np.ndarray) -> None:
        """Compute the total momentum (inclusive of kz) at different binding energies."""
        self.k_tot = _safe_compute_k_tot(
            self.ds.S.hv, self.ds.S.work_function, binding_energy
        )

    def kspace_to_phi(
        self, binding_energy: np.ndarray, kp: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """Converts from momentum back to the analyzer angular axis."""
        if self.phi is not None:
            return self.phi

        if self.k_tot is None:
            self.compute_k_tot(binding_energy)

        self.phi = np.zeros_like(kp)
        par_tot = isinstance(self.k_tot, np.ndarray) and len(self.k_tot) != 1
        assert len(self.k_tot) == len(kp) or len(self.k_tot) == 1

        _small_angle_arcsin(
            kp / np.cos(self.perpendicular_offset),
            self.k_tot,
            self.phi,
            self.phi_offset,
            par_tot,
            False,
        )

        if self.calibration is not None:
            self.phi = self.calibration.correct_detector_angle(
                eV=binding_energy, phi=self.phi
            )
        return self.phi

    def conversion_for(self, dim: str) -> Callable:
        """Looks up the appropriate momentum-to-angle conversion routine by dimension name."""

        def with_identity(*args, **kwargs):
            return self.identity_transform(dim, *args, **kwargs)

        return {"eV": self.kspace_to_BE, "phi": self.kspace_to_phi}.get(
            dim, with_identity
        )


class ConvertKxKy(CoordinateConverter):
    """Implements volumetric momentum conversion for kx-ky scans.

    Please note that currently we assume that psi = 0 when you are not using an
    electrostatic deflector.
    """

    k_tot: np.ndarray = None
    phi: np.ndarray = None
    perp_angle: np.ndarray = None
    rkx: np.ndarray = None
    rky: np.ndarray = None

    def __init__(self, ds: xr.Dataset, *args: list[str], **kwargs: Any) -> None:
        """Initialize the kx-ky momentum converter and cached coordinate values."""
        super().__init__(ds, *args, **kwargs)
        # accept either vertical or horizontal, fail otherwise
        if not any(
            np.abs(ds.alpha - alpha_option) < (np.pi / 180)
            for alpha_option in [0, np.pi / 2]
        ):
            raise ValueError(
                "You must convert either vertical or horizontal slit data with this "
                "converter."
            )

        self.direct_angles = (
            "phi",
            [d for d in ["psi", "beta", "theta"] if d in ds.indexes][0],
        )
        assert len(self.direct_angles) == 2, "couldn't find an appropriate scan angle"

        if self.direct_angles[1] != "psi":
            # psi allows for either orientation
            assert (
                self.direct_angles[1] == "beta"
            ) != self.is_slit_vertical, "slit orientation doesn't match scan angle"

        if self.is_slit_vertical:
            self.parallel_angles = (
                "beta",
                "psi" if "theta" in self.direct_angles else "theta",
            )
        else:
            self.parallel_angles = (
                "theta",
                "psi" if "beta" in self.direct_angles else "beta",
            )

    @property
    def phi_offset(self) -> float:
        """The offset for the phi angle."""
        return self.ds.S.phi_offset + self.ds.S.lookup_offset_coord(
            self.parallel_angles[0]
        )

    @property
    def perp_offset(self) -> float:
        """The offset for the perpendicular angle."""
        scan_angle = self.direct_angles[1]
        return self.ds.S.lookup_offset(scan_angle) + self.ds.S.lookup_offset_coord(
            self.parallel_angles[1]
        )

    def get_coordinates(
        self, resolution: dict = None, bounds: dict = None
    ) -> dict[str, np.ndarray]:
        """Calculates appropriate coordinate bounds."""
        resolution = {} if resolution is None else resolution
        bounds = {} if bounds is None else bounds

        coordinates = super().get_coordinates(resolution, bounds=bounds)

        kx_angle, ky_angle = self.direct_angles
        ((kx_low, kx_high), (ky_low, ky_high)) = calculate_kx_ky_bounds(
            self.ds, self.phi_offset, self.perp_offset, ky_angle
        )

        if "kx" in bounds:
            kx_low, kx_high = bounds["kx"]
        if "ky" in bounds:
            ky_low, ky_high = bounds["ky"]

        len_ky_angle = len(self.ds.coords[ky_angle])
        len_kx_angle = len(self.ds.coords[kx_angle])

        inferred_kx_res = (kx_high - kx_low + 2 * K_SPACE_BORDER) / len_kx_angle
        inferred_ky_res = (ky_high - ky_low + 2 * K_SPACE_BORDER) / len_ky_angle
        # upsample a bit if there aren't that many points along a certain axis
        try:
            inferred_kx_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kx_res][
                -2 if (len_kx_angle < 80) else -1
            ]
        except IndexError:
            inferred_kx_res = MOMENTUM_BREAKPOINTS[-2]
        try:
            inferred_ky_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_ky_res][
                -2 if (len_ky_angle < 80) else -1
            ]
        except IndexError:
            inferred_ky_res = MOMENTUM_BREAKPOINTS[-2]

        coordinates["kx"] = np.arange(
            kx_low - K_SPACE_BORDER,
            kx_high + K_SPACE_BORDER,
            resolution.get("kx", inferred_kx_res),
        )
        coordinates["ky"] = np.arange(
            ky_low - K_SPACE_BORDER,
            ky_high + K_SPACE_BORDER,
            resolution.get("ky", inferred_ky_res),
        )

        # TODO J: figure out a better way to handle this
        # J: turns out this is required even while keeping the original dataset
        base_coords = {
            k: v
            for k, v in self.ds.coords.items()
            if k not in ["eV", "phi", "psi", "theta", "beta", "alpha", "chi"]
        }
        coordinates.update(base_coords)

        return coordinates

    def compute_k_tot(self, binding_energy: np.ndarray) -> None:
        """Compute the total momentum (inclusive of kz) at different binding energies."""
        self.k_tot = _safe_compute_k_tot(
            self.ds.S.hv, self.ds.S.work_function, binding_energy
        )

    def conversion_for(self, dim: str) -> Callable:
        """Looks up the appropriate momentum-to-angle conversion routine by dimension name."""

        def with_identity(*args, **kwargs):
            return self.identity_transform(dim, *args, **kwargs)

        return {
            "eV": self.kspace_to_BE,
            "phi": self.kspace_to_phi,
            "theta": self.kspace_to_perp_angle,
            "psi": self.kspace_to_perp_angle,
            "beta": self.kspace_to_perp_angle,
        }.get(dim, with_identity)

    @property
    def needs_rotation(self) -> bool:
        """Whether we need to rotate the momentum coordinates when converting to angle."""
        # force rotation when greater than 0.5 deg
        return np.abs(self.ds.S.lookup_offset_coord("chi")) > (0.5 * np.pi / 180)

    def rkx_rky(self, kx, ky):
        """Returns the rotated kx and ky values when we are rotating by nonzero chi."""
        if self.rkx is not None:
            return self.rkx, self.rky

        chi = self.ds.S.lookup_offset_coord("chi")

        self.rkx = np.zeros_like(kx)
        self.rky = np.zeros_like(ky)
        _rotate_kx_ky(kx, ky, self.rkx, self.rky, chi)

        return self.rkx, self.rky

    def kspace_to_phi(
        self,
        binding_energy: np.ndarray,
        kx: np.ndarray,
        ky: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """Converts from momentum back to the analyzer angular axis."""
        if self.phi is not None:
            return self.phi

        if self.k_tot is None:
            self.compute_k_tot(binding_energy)

        if self.needs_rotation:
            kx, ky = self.rkx_rky(kx, ky)

        # This can be condensed but it is actually better not to condense it:
        # In this format, we can very easily compare to the raw coordinate conversion functions that
        # come from Mathematica in order to adjust signs, etc.
        self.phi = np.zeros_like(ky)

        par_tot = isinstance(self.k_tot, np.ndarray) and len(self.k_tot) != 1
        assert len(self.k_tot) == len(self.phi) or len(self.k_tot) == 1

        _exact_arcsin(kx, ky, self.k_tot, self.phi, self.phi_offset, par_tot, False)

        if self.calibration is not None:
            self.phi = self.calibration.correct_detector_angle(
                eV=binding_energy, phi=self.phi
            )

        return self.phi

    def kspace_to_perp_angle(
        self,
        binding_energy: np.ndarray,
        kx: np.ndarray,
        ky: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """Convert momentum to the scan angle perpendicular to the analyzer slit."""
        if self.perp_angle is not None:
            return self.perp_angle

        if self.k_tot is None:
            self.compute_k_tot(binding_energy)

        if self.needs_rotation:
            kx, ky = self.rkx_rky(kx, ky)

        self.perp_angle = np.zeros_like(kx)

        par_tot = isinstance(self.k_tot, np.ndarray) and len(self.k_tot) != 1
        assert len(self.k_tot) == len(self.perp_angle) or len(self.k_tot) == 1

        _exact_arcsin(
            ky, kx, self.k_tot, self.perp_angle, self.perp_offset, par_tot, False
        )

        return self.perp_angle
