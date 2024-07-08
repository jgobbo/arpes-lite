"""Provides coordinate aware filters and smoothing."""

import copy

import numpy as np
from scipy import ndimage

import xarray as xr
from arpes.provenance import provenance
from arpes.utilities import lift_spectrum

__all__ = (
    "gaussian_filter_arr",
    "gaussian_filter",
    "boxcar_filter_arr",
    "boxcar_filter",
)


@lift_spectrum
def gaussian_filter_arr(
    spectrum: xr.DataArray, sigma=None, n=1, default_size=1
) -> xr.DataArray:
    """Coordinate aware `scipy.ndimage.filters.gaussian_filter`.

    Args:
        data: Data to smooth.
        sigma: Kernel sigma, specified in terms of axis units. An axis that is not specified
          will have a kernel width of `default_size` in index units.
        n: Repeats n times.
        default_size: Changes the default kernel width for axes not specified in `sigma`. Changing this
          parameter and leaving `sigma` as None allows you to smooth with an even-width
          kernel in index-coordinates.

    Returns:
        Smoothed data.
    """

    sigma = {} if sigma is None else sigma
    sigma = {
        k: int(v / (spectrum.coords[k][1] - spectrum.coords[k][0]))
        for k, v in sigma.items()
    }
    for dim in spectrum.dims:
        if dim not in sigma:
            sigma[dim] = default_size
    sigma = tuple(sigma[k] for k in spectrum.dims)

    values = spectrum.values
    for _ in range(n):
        values = ndimage.filters.gaussian_filter(values, sigma)

    filtered_spectrum = xr.DataArray(values, spectrum.coords, spectrum.dims)
    return filtered_spectrum


def gaussian_filter(sigma=None, n=1):
    """A partial application of `gaussian_filter_arr` that can be passed to derivative analysis functions.

    Args:
        sigma
        n

    Returns:
        A function which applies the Gaussian filter.
    """

    def f(arr):
        return gaussian_filter_arr(arr, sigma, n)

    return f


def boxcar_filter(size=None, n=1):
    """A partial application of `boxcar_filter_arr` that can be passed to derivative analysis functions.

    Args:
        size
        n

    Returns:
        A function which applies the boxcar.
    """

    def f(arr):
        return boxcar_filter_arr(arr, size, n)

    return f


def boxcar_filter_arr(
    arr: xr.DataArray, size=None, n=1, default_size=1, skip_nan=True
) -> xr.DataArray:
    """Coordinate aware `scipy.ndimage.filters.boxcar_filter`.

    Args:
        arr
        size: Kernel size, specified in terms of axis units. An axis
            that is not specified will have a kernel width of
            `default_size` in index units.
        n: Repeats n times.
        default_size: Changes the default kernel width for axes not
            specified in `sigma`. Changing this parameter and leaving
            `sigma` as None allows you to smooth with an even-width
            kernel in index-coordinates.
        skip_nan: By default, masks parts of the data which are NaN to
            prevent poor filter results.

    Returns:
        smoothed data.
    """
    if size is None:
        size = {}

    size = {k: int(v / (arr.coords[k][1] - arr.coords[k][0])) for k, v in size.items()}
    for dim in arr.dims:
        if dim not in size:
            size[dim] = default_size

    size = tuple(size[k] for k in arr.dims)

    if skip_nan:
        nan_mask = np.copy(arr.values) * 0 + 1
        nan_mask[arr.values != arr.values] = 0
        filtered_mask = ndimage.filters.uniform_filter(nan_mask, size)

        values = np.copy(arr.values)
        values[values != values] = 0

        for _ in range(n):
            values = ndimage.filters.uniform_filter(values, size) / filtered_mask
            values[nan_mask == 0] = 0
    else:
        for i in range(n):
            values = ndimage.filters.uniform_filter(values, size)

    filtered_arr = xr.DataArray(
        values, arr.coords, arr.dims, attrs=copy.deepcopy(arr.attrs)
    )

    if "id" in arr.attrs:
        del filtered_arr.attrs["id"]

        provenance(
            filtered_arr,
            arr,
            {
                "what": "Boxcar filtered data",
                "by": "boxcar_filter_arr",
                "size": size,
                "skip_nan": skip_nan,
            },
        )

    return filtered_arr
