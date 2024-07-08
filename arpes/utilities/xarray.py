"""Utilities related to function application on xr types."""

import numpy as np
import xarray as xr

from typing import Callable, Any
from functools import wraps

from arpes.typing import DataType

__all__ = (
    "apply_dataarray",
    "lift_datavar_attrs",
    "lift_dataarray_attrs",
    "lift_dataarray",
    "lift_spectrum",
    "unwrap_xarray_item",
    "unwrap_xarray_dict",
    "rename_dataset_keys",
)


def unwrap_xarray_item(item):
    """Unwraps something that might or might not be an xarray like with .item() attribute.

    This is especially helpful for dealing with unwrapping coordinates which might
    be floating point-like or might be array-like.

    Args:
        item: The value to unwrap.

    Returns:
        The safely unwrapped item
    """
    try:
        return item.item()
    except (AttributeError, ValueError):
        return item


def unwrap_xarray_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Returns the attributes as unwrapped values rather than item() instances.

    Useful for unwrapping coordinate dicts where the values might be a bare type:
    like a float or an int, but might also be a wrapped array-like for instance
    xr.DataArray. Even worse, we might have wrapped bare values!

    Args:
        d

    Returns:
        The unwrapped attributes as a dict.
    """
    return {k: unwrap_xarray_item(v) for k, v in d.items()}


def apply_dataarray(arr: xr.DataArray, f, *args, **kwargs):
    """Applies a function onto the values of a DataArray."""
    return xr.DataArray(
        f(arr.values, *args, **kwargs), arr.coords, arr.dims, attrs=arr.attrs
    )


def lift_dataarray(
    f: Callable[[np.ndarray], np.ndarray]
) -> Callable[[xr.DataArray], xr.DataArray]:
    """Lifts a function that operates on an np.ndarray's values to act on an xr.DataArray.

    Args:
        f

    Returns:
        g: Function operating on an xr.DataArray
    """

    @wraps(f)
    def g(arr: xr.DataArray, *args, **kwargs):
        return apply_dataarray(arr, f, *args, **kwargs)

    return g


def lift_dataarray_attrs(
    f: Callable[[dict], dict]
) -> Callable[[xr.DataArray], xr.DataArray]:
    """Lifts a function that operates dicts to a function that acts on dataarray attrs.

    Produces a new xr.DataArray.

    Args:
        f

    Returns:
        g: Function operating on the attributes of an xr.DataArray
    """

    @wraps(f)
    def g(arr: xr.DataArray, *args, **kwargs):
        return xr.DataArray(
            arr.values, arr.coords, arr.dims, attrs=f(arr.attrs, *args, **kwargs)
        )

    return g


def lift_datavar_attrs(f: Callable[[dict], dict]) -> Callable[[DataType], DataType]:
    """
    Lifts a function that operates dicts to a function that acts on xr attrs.

    Applies to all attributes of all the datavars in a xr.Dataset, as well as the
    Dataset attrs themselves.

    Args:
        f: Function to apply

    Returns:
        The function modified to apply to xr instances.
    """

    @wraps(f)
    def g(data: DataType, *args, **kwargs):
        arr_lifted = lift_dataarray_attrs(f)
        if isinstance(data, xr.DataArray):
            return arr_lifted(data, *args, **kwargs)

        var_names = list(data.data_vars.keys())
        new_vars = {k: arr_lifted(data[k], *args, **kwargs) for k in var_names}
        new_root_attrs = f(data.attrs, *args, **kwargs)

        return xr.Dataset(new_vars, data.coords, new_root_attrs)

    return g


def lift_spectrum(function: Callable) -> Callable:
    """
    If the decorated function is supplied a Dataset by the user, this will lift the
    spectrum DataArray and set it as the first argument to the function. The decorated
    functions first argument must be a DataArray.

    Args:
        function: The wrapped function
    """

    @wraps(function)
    def with_dataarray(data: DataType, *args, **kwargs):
        if isinstance(data, xr.Dataset):
            spectrum = data.S.spectrum
            return function(spectrum, *args, **kwargs)
        assert isinstance(data, xr.DataArray), "Data must be a Dataset or DataArray."
        return function(data, *args, **kwargs)

    return with_dataarray


def rename_dataset_keys(dataset: xr.Dataset, rename_keys: dict):
    """
    Renames all keys in a dataset, including data variables, coordinates, dimensions,
    and attributes.
    """
    existing_data_keys = set.union(
        *[set(key_dict.keys()) for key_dict in [dataset.coords, dataset.data_vars]]
    )
    existing_attr_keys = set(dataset.attrs.keys())

    renameable_data = {}
    renameable_attrs = {}
    for key in rename_keys.keys():
        # J: We prefer to rename data over attributes. We may want to swap this or rename both.
        if key in existing_data_keys:
            renameable_data[key] = rename_keys[key]
        elif key in existing_attr_keys:
            renameable_attrs[key] = rename_keys[key]

    dataset = dataset.rename(renameable_data)
    for k, v in renameable_attrs.items():
        dataset.attrs[v] = dataset.attrs.pop(k)

    return dataset
