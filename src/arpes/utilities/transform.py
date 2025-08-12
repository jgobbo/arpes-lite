"""Collection of functions to transform data."""

from scipy.ndimage import rotate
from arpes.utilities import lift_spectrum

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xarray import DataArray


@lift_spectrum
def rotate_spectrum(spectrum: "DataArray", angle: float) -> "DataArray":
    """Rotate a 2D array by a given `angle`.

    Uses scipys `rotate` function and combines the result with coordinates calculated
    from the original array. It doesn't make sense to use this function

    Args
    ----
    spectrum : DataArray
        The 2D array to rotate.
    angle : float
        The angle to rotate the array by in degrees.
    """

    assert spectrum.ndim == 2, "Can only rotate 2D arrays"

    array = spectrum.values
    rotated_array = rotate(array, angle, reshape=False)
