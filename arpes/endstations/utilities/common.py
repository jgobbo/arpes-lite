from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def safe_reshape(data: "np.ndarray", shape: tuple[int]):
    """
    Attempt to reshape data to given shape. If the data is shorter than the shape, the
    final dimension will be shorter than requested. This is often useful for reshaping
    data from a scan that was terminated early.

    Args:
        data: The data to reshape
        shape: The new shape

    Returns:
        reshaped: the reshaped data
    """
    new_shape = shape[:-1] + (-1,)
    return data.reshape(new_shape)
