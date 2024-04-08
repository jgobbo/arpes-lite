"""Automated utilities for calculating Fermi edge corrections."""

import lmfit as lf
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

import xarray as xr
from arpes.constants import K_BOLTZMANN_EV_KELVIN
from arpes.fits import (
    GStepBModel,
    LinearModel,
    QuadraticModel,
    AffineBroadenedFD,
    broadcast_model,
)
from arpes.provenance import provenance, update_provenance
from arpes.utilities.math import shift_by


def _exclude_from_set(excluded):
    def exclude(l):
        return list(set(l).difference(excluded))

    return exclude


exclude_hemisphere_axes = _exclude_from_set({"phi", "eV"})
exclude_hv_axes = _exclude_from_set({"hv", "eV"})


__all__ = (
    "build_quadratic_fermi_edge_correction",
    "build_photon_energy_fermi_edge_correction",
    "apply_photon_energy_fermi_edge_correction",
    "apply_quadratic_fermi_edge_correction",
    "build_direct_fermi_edge_correction",
    "apply_direct_fermi_edge_correction",
    "find_e_fermi_linear_dos",
    "fix_fermi_edge",
)


def find_e_fermi_linear_dos(edc, guess=None, plot=False, ax=None):
    """Estimate the Fermi level under the assumption of a linear density of states.

    Does a reasonable job of finding E_Fermi in-situ for graphene/graphite or other materials with a linear DOS near
    the chemical potential. You can provide an initial guess via guess, or one will be chosen half way through the EDC.

    The Fermi level is estimated as the location where the DoS crosses below an estimated background level

    Args:
        edc: Input data
        guess: Approximate location
        plot: Whether to plot the fit, useful for debugging.

    Returns:
        The Fermi edge position.
    """
    if guess is None:
        guess = edc.eV.values[len(edc.eV) // 2]

    edc = edc - np.percentile(edc.values, (20,))[0]
    mask = edc > np.percentile(edc.sel(eV=slice(None, guess)), 20)
    mod = LinearModel().guess_fit(edc[mask])

    chemical_potential = -mod.params["intercept"].value / mod.params["slope"].value

    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        edc.plot(ax=ax)
        ax.axvline(chemical_potential, linestyle="--", color="red")
        ax.axvline(guess, linestyle="--", color="gray")

    return chemical_potential


def apply_direct_fermi_edge_correction(
    arr: xr.DataArray, correction=None, *args, **kwargs
):
    """Applies a direct fermi edge correction stencil."""
    if correction is None:
        correction = build_direct_fermi_edge_correction(arr, *args, **kwargs)

    shift_amount = (
        -correction / arr.G.stride(generic_dim_names=False)["eV"]
    )  # pylint: disable=invalid-unary-operand-type
    energy_axis = list(arr.dims).index("eV")

    correction_axis = list(arr.dims).index(correction.dims[0])

    corrected_arr = xr.DataArray(
        shift_by(
            arr.values, shift_amount, axis=energy_axis, by_axis=correction_axis, order=1
        ),
        arr.coords,
        arr.dims,
        attrs=arr.attrs,
    )

    if "id" in corrected_arr.attrs:
        del corrected_arr.attrs["id"]

    provenance(
        corrected_arr,
        arr,
        {
            "what": "Shifted Fermi edge to align at 0 along hv axis",
            "by": "apply_photon_energy_fermi_edge_correction",
            "correction": list(
                correction.values
                if isinstance(correction, xr.DataArray)
                else correction
            ),
        },
    )

    return corrected_arr


@update_provenance("Build direct Fermi edge correction")
def build_direct_fermi_edge_correction(
    arr: xr.DataArray, fit_limit=0.001, energy_range=None, plot=False, along="phi"
):
    """Builds a direct fermi edge correction stencil.

    This means that fits are performed at each value of the 'phi' coordinate
    to get a list of fits. Bad fits are thrown out to form a stencil.

    This can be used to shift coordinates by the nearest value in the stencil.

    Args:
        arr
        fit_limit
        energy_range
        plot
        along

    Returns:
        The array of fitted edge coordinates.
    """
    if energy_range is None:
        energy_range = slice(-0.1, 0.1)

    exclude_axes = ["eV", along]
    others = [d for d in arr.dims if d not in exclude_axes]
    edge_fit = broadcast_model(
        GStepBModel, arr.sum(others).sel(eV=energy_range), along
    ).results

    def sieve(c, v):
        return v.item().params["center"].stderr < 0.001

    corrections = edge_fit.G.filter_coord(along, sieve).G.map(
        lambda x: x.params["center"].value
    )

    if plot:
        corrections.plot()

    return corrections


def build_quadratic_fermi_edge_correction(
    arr: xr.DataArray, fit_limit=0.001, eV_slice=None, plot=False
) -> lf.model.ModelResult:
    """Calculates a quadratic Fermi edge correction by edge fitting and then quadratic fitting of edges."""
    # TODO improve robustness here by allowing passing in the location of the fermi edge guess
    # We could also do this automatically by using the same method we use for step detection to find the edge of the
    # spectrometer image

    if eV_slice is None:
        approximate_fermi_level = arr.S.find_spectrum_energy_edges().max()
        eV_slice = slice(approximate_fermi_level - 0.4, approximate_fermi_level + 0.4)
    else:
        approximate_fermi_level = 0
    sum_axes = exclude_hemisphere_axes(arr.dims)
    edge_fit = broadcast_model(
        GStepBModel,
        arr.sum(sum_axes).sel(eV=eV_slice),
        "phi",
        params={"center": {"value": approximate_fermi_level}},
    )

    size_phi = len(arr.coords["phi"])
    not_nanny = (np.logical_not(np.isnan(arr)) * 1).sum("eV") > size_phi * 0.30
    condition = np.logical_and(edge_fit.F.s("center") < fit_limit, not_nanny)

    quadratic_corr = QuadraticModel().guess_fit(
        edge_fit.F.p("center"), weights=condition * 1
    )
    if plot:
        edge_fit.F.p("center").plot()
        plt.plot(arr.coords["phi"], quadratic_corr.best_fit)

    return quadratic_corr


@update_provenance("Build photon energy Fermi edge correction")
def build_photon_energy_fermi_edge_correction(
    arr: xr.DataArray, plot=False, energy_window=0.2
):
    """Builds Fermi edge corrections across photon energy (corrects monochromator miscalibration)."""
    edge_fit = broadcast_model(
        GStepBModel,
        arr.sum(exclude_hv_axes(arr.dims)).sel(eV=slice(-energy_window, energy_window)),
        "hv",
    )

    return edge_fit


def apply_photon_energy_fermi_edge_correction(
    arr: xr.DataArray, correction=None, **kwargs
):
    """Applies Fermi edge corrections across photon energy (corrects monochromator miscalibration)."""
    if correction is None:
        correction = build_photon_energy_fermi_edge_correction(arr, **kwargs)

    correction_values = correction.G.map(lambda x: x.params["center"].value)
    if "corrections" not in arr.attrs:
        arr.attrs["corrections"] = {}

    arr.attrs["corrections"]["hv_correction"] = list(correction_values.values)

    shift_amount = -correction_values / arr.G.stride(generic_dim_names=False)["eV"]
    energy_axis = arr.dims.index("eV")
    hv_axis = arr.dims.index("hv")

    corrected_arr = xr.DataArray(
        shift_by(arr.values, shift_amount, axis=energy_axis, by_axis=hv_axis, order=1),
        arr.coords,
        arr.dims,
        attrs=arr.attrs,
    )

    if "id" in corrected_arr.attrs:
        del corrected_arr.attrs["id"]

    provenance(
        corrected_arr,
        arr,
        {
            "what": "Shifted Fermi edge to align at 0 along hv axis",
            "by": "apply_photon_energy_fermi_edge_correction",
            "correction": list(correction_values.values),
        },
    )

    return corrected_arr


def apply_quadratic_fermi_edge_correction(
    arr: xr.DataArray, correction: lf.model.ModelResult = None, offset=None
):
    """Applies a Fermi edge correction using a quadratic fit for the edge."""
    assert isinstance(arr, xr.DataArray)
    if correction is None:
        correction = build_quadratic_fermi_edge_correction(arr)

    if "corrections" not in arr.attrs:
        arr.attrs["corrections"] = {}

    arr.attrs["corrections"]["FE_Corr"] = correction.best_values

    delta_E = arr.coords["eV"].values[1] - arr.coords["eV"].values[0]
    dims = list(arr.dims)
    energy_axis = dims.index("eV")
    phi_axis = dims.index("phi")

    shift_amount_E = correction.eval(x=arr.coords["phi"].values)

    if offset is not None:
        shift_amount_E = shift_amount_E - offset

    shift_amount = -shift_amount_E / delta_E

    corrected_arr = xr.DataArray(
        shift_by(arr.values, shift_amount, axis=energy_axis, by_axis=phi_axis, order=1),
        arr.coords,
        arr.dims,
        attrs=arr.attrs,
    )

    if "id" in corrected_arr.attrs:
        del corrected_arr.attrs["id"]

    provenance(
        corrected_arr,
        arr,
        {
            "what": "Shifted Fermi edge to align at 0",
            "by": "apply_quadratic_fermi_edge_correction",
            "correction": correction.best_values,
        },
    )

    return corrected_arr


# def fd(E, amp, Ef, T):
#     """
#     Returns a Fermi-Dirac distribution on the input E values.
#     Accepts (outputs) E (fermi dirac) as numpy array, xarray DataArray, or xarray DataSet
#     Ef and E in eV, T in K
#     """
#     numerator = E - Ef
#     denominator = K_BOLTZMANN_EV_KELVIN * T

#     return amp / (np.exp(numerator / denominator) + 1)


# def find_fermi_edge(da, guess_Ef=(-0.1, 0.1), bin=5):
#     """
#     Description: Identifies Fermi edge from a 2D or 3D E-K spectrum by fitting a Fermi-Dirac distribution to
#     all EDCs.
#     Inputs:
#     - da: (DataArray) 2D or 3D spectrum of E-k or E-kx-ky
#     - guess_Ef: (tuple of floats) Range of E values to look for Fermi edge
#     - bin: (int or list/tuple of ints) Number of elements to use as sigma for gaussian blurring in non-E dimensions.
#     For data with multiple non-E coordinates, an int or sequence of ints can be supplied
#     Output:
#     (DataArray) Array with one less dimension than da, with fitted Ef values over momentum coordinates
#     Beware: Uses fd function to fit to. Needs gaussian_filter to be imported from scipy.ndimage
#     """

#     # Start with fermi edge fitting using FD distribution

#     sig = np.concatenate(([0], np.ones(len(da.dims) - 1) * bin))

#     # blur in momentum direction for smoother fermi edge

#     data = da.sel(eV=slice(*guess_Ef)).copy(
#         data=gaussian_filter(da.sel(eV=slice(*guess_Ef)), sigma=sig), deep=True
#     )

#     guess_params = {"amp": np.mean(data.values) * 2, "Ef": np.mean(guess_Ef), "T": 50}

#     fit = data.curvefit("eV", fd, p0=guess_params, bounds={"T": (1, 500)})

#     return fit["curvefit_coefficients"].sel(param="Ef")


# def fermi_edge_correction(da, fermi_edge):
#     """
#     Shifts band structure (E-K plot) along E axis by momentum-dependent amount. If shift amount is chosen as
#     the fermi edge energies, this function outputs the spectrum with corrected fermi edge shape.
#     Inputs:
#     - da: (DataArray) Array containing uncorrected 2D or 3D band structure data (E-k or E-kx-ky).
#     - fermi_edge: (DataArray) Array of one less dimension than da, with values equal to fermi edge energies
#     in da, with identical non-E coordinates. For 3D da, if a 1D fermi_edge is supplied the function will
#     copy the 1D fermi_edge over the remaining coordinate.
#     Output: DataArray identical to input, but with each EDC shifted by amounts given by
#     fermi_edge-np.max(fermi_edge). If fermi_edge is the uncorrected fermi level, the output is a spectrum
#     with fermi level equal to the peak of the uncorrected fermi edge, independent of all non-E coordinates.
#     Beware: assumes eV is the first dimension in da.
#     """

#     eVrange = np.abs(da["eV"][-1].values - da["eV"][0].values)

#     idxshift = np.outer(
#         np.ones(len(da["eV"])),
#         (fermi_edge - np.max(fermi_edge)).values * len(da["eV"]) / eVrange,
#     )

#     da_ft = np.fft.fft(da.data, axis=0)

#     if len(da.dims) == 2:
#         ones_mat = np.ones(len(da[da.dims[1]]))

#     else:
#         if len(da.dims) == 3:
#             ones_mat = np.ones((len(da[da.dims[1]]), len(da[da.dims[2]])))

#             idxshift = np.outer(idxshift, np.ones(len(da[da.dims[2]])))

#     exp_mat = np.exp(
#         -1j * 2 * np.pi * np.multiply.outer(np.arange(len(da["eV"])), ones_mat) / len(da["eV"])
#     )

#     return da.copy(data=np.real(np.fft.ifft(exp_mat ** (-idxshift) * da_ft, axis=0)), deep=True)


# def fast_fix_one_fermi_edge(cut: xr.DataArray):
#     edge = find_fermi_edge(cut)
#     return fermi_edge_correction(cut, edge)


def fix_one_fermi_edge(cut: xr.DataArray, edge_fit=None):
    edge_fit = edge_fit if edge_fit is not None else fit_fermi_edge(cut)
    return cut.G.shift_by(edge_fit, "eV")


def fit_fermi_edge(cut: xr.DataArray):
    edge_region = cut.sel(eV=slice(-0.1, 0.1))

    edge_fits = broadcast_model(AffineBroadenedFD, edge_region, "phi", progress=False)
    quad_fit = QuadraticModel().guess_fit(edge_fits.F.p("fd_center"))
    return quad_fit.eval(x=edge_region["phi"])


def fix_fermi_edge(ds: xr.Dataset, broadcast_fit: bool = True) -> xr.Dataset:
    """
    Automatically corrects the Fermi edge in a dataset. If the measurement was done with a
    curved slit, the Fermi edge will only be shifted to 0. If the measurement was done with a
    straight slit, the Fermi edge will be shifted to 0 and the curved Fermi edge will be corrected.

    Args:
        ds: The dataset to correct
        broadcast_fit: Whether to fit the Fermi edge of the summed spectrum and then broadcast to
            all spectra or fit each spectrum individually.

    Returns:
        A new dataset with the Fermi edge corrected.
    """
    HEMISPHERE_DIMS = {"eV", "phi"}
    ds = ds.copy(deep=True)
    curved_edge = ds.attrs["slit_shape"] == "straight"
    spectrum = ds.S.spectrum

    if not all(dim in spectrum.dims for dim in HEMISPHERE_DIMS):
        return ds

    # Moving the Fermi edge to 0
    integrated_edc = spectrum.sum([dim for dim in spectrum.dims if dim != "eV"])
    dropoff_index = np.argmin(np.diff(integrated_edc))
    zero_index = np.argmin(np.abs(spectrum["eV"].values))
    spectrum = spectrum.shift(eV=zero_index - dropoff_index)
    if curved_edge is False:
        ds = ds.assign_coords({"eV_shifted": ds["eV"] - ds["eV"].values[dropoff_index]})
        ds = ds.assign_coords({"eV": ds["eV_shifted"]})
        ds = ds.drop_vars("eV_shifted")
        return ds

    scan_axes = [dim for dim in spectrum.dims if dim not in HEMISPHERE_DIMS]
    stacked = spectrum.stack(flattened=scan_axes)
    edge_fit = (
        fit_fermi_edge(stacked.sum("flattened")) if broadcast_fit is True else None
    )

    fixed_cuts = [
        fix_one_fermi_edge(cut, edge_fit) for cut in stacked.transpose("flattened", ...)
    ]
    fixed_stack = xr.concat(fixed_cuts, dim="flattened")
    stacked.values = fixed_stack.transpose(*stacked.dims)

    return ds.update({ds.S.spectrum_name: stacked.unstack("flattened")})
