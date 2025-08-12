"""Utilities and an example of how to make an animated plot to export as a movie."""

from pathlib import Path
from warnings import warn
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation

import xarray as xr
from arpes.utilities import lift_spectrum

__all__ = ("plot_movie",)


@lift_spectrum
def plot_movie(
    spectrum: xr.DataArray,
    time_dim: str,
    framerate: int = 10,
    ax=None,
    save_path: Path | str | None = None,
    **kwargs,
):
    """Make an animated plot of a 3D dataset using one dimension as `time`.

    Args
    ----
    spectrum : xr.DataArray
        The 3D dataset to animate.
    time_dim : str
        The name of the dimension to animate over.
    framerate : int (default, 10)
        The number of frames per second.
    ax : plt.Axes (default, None)
        The axes to plot on. If None, a new figure is created.
    save_path : Path | str | None (default, None)
        The path to save the movie to. If None, the movie is not saved.

    Returns
    -------
    animation : FuncAnimation
        The animation object. If `save_path` is not None, the movie is saved to that
        path.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.get_figure()

    vmax = spectrum.max().item()
    vmin = spectrum.min().item()

    cmap = plt.rcParams["image.cmap"]
    if spectrum.S.is_subtracted:
        cmap = "RdBu"
        vmax = np.max([np.abs(vmin), np.abs(vmax)])
        vmin = -vmax
    vmax = kwargs.get("vmax", vmax)
    vmin = kwargs.get("vmin", vmin)

    def animate(i):
        ax.clear()
        plot = spectrum.isel(**{time_dim: i}).plot(ax=ax, add_colorbar=False)
        return plot

    animation = FuncAnimation(
        fig,
        animate,
        frames=spectrum.sizes[time_dim],
        interval=1000 / framerate,
        **kwargs,
    )

    if save_path is not None:
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        valid_extension = ".gif"
        if save_path.suffix != valid_extension:
            save_path = save_path.with_suffix(valid_extension)
            warn(
                f"save_path should have a {valid_extension} extension. Saving to {save_path} "
                "instead."
            )
        writer = PillowWriter(fps=framerate)
        animation.save(save_path, writer=writer)

    return animation
