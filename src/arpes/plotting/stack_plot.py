"""Plotting routines for making the classic stacked line plots.

Think the album art for "Unknown Pleasures".
"""

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import xarray as xr
from arpes.analysis.general import rebin
from arpes.plotting.utils import (
    colorbarmaps_for_axis,
    generic_colorbarmap_for_data,
    fancy_labels,
    path_for_plot,
    label_for_dim,
)
from arpes.provenance import save_plot_provenance
from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum, lift_spectrum

__all__ = (
    "stack_dispersion_plot",
    "flat_stack_plot",
    "overlapped_stack_dispersion_plot",
)


@save_plot_provenance
def flat_stack_plot(
    data: DataType,
    stack_axis=None,
    fermi_level=True,
    cbarmap=None,
    ax=None,
    mode="line",
    title=None,
    out=None,
    transpose=False,
    **kwargs,
):
    """Generates a stack plot with all the lines distinguished by color rather than offset."""
    data = normalize_to_spectrum(data)
    if len(data.dims) != 2:
        raise ValueError(
            "In order to produce a stack plot, data must be image-like."
            "Passed data included dimensions: {}".format(data.dims)
        )

    fig = None
    inset_ax = None
    if ax is None:
        fig, ax = plt.subplots(
            figsize=kwargs.get(
                "figsize",
                (
                    7,
                    5,
                ),
            )
        )
        inset_ax = inset_axes(ax, width="40%", height="5%", loc=1)

    if stack_axis is None:
        stack_axis = data.dims[0]

    skip_colorbar = True
    if cbarmap is None:
        skip_colorbar = False
        try:
            cbarmap = colorbarmaps_for_axis[stack_axis]
        except KeyError:
            cbarmap = generic_colorbarmap_for_data(
                data.coords[stack_axis], ax=inset_ax, ticks=kwargs.get("ticks")
            )

    cbar, cmap = cbarmap

    # should be exactly two
    other_dim = [d for d in data.dims if d != stack_axis][0]
    other_coord = data.coords[other_dim]

    if not isinstance(cmap, matplotlib.colors.Colormap):
        # do our best
        try:
            cmap = cmap()
        except:
            # might still be fine
            pass

    if "eV" in data.dims and "eV" != stack_axis and fermi_level:
        if transpose:
            ax.axhline(0, color="red", alpha=0.8, linestyle="--", linewidth=1)
        else:
            ax.axvline(0, color="red", alpha=0.8, linestyle="--", linewidth=1)

    # meat of the plotting
    for coord_dict, marginal in list(data.G.iterate_axis(stack_axis)):
        if transpose:
            if mode == "line":
                ax.plot(
                    marginal.values,
                    marginal.coords[marginal.dims[0]].values,
                    color=cmap(coord_dict[stack_axis]),
                    **kwargs,
                )
            else:
                assert mode == "scatter"
                raise NotImplementedError
        else:
            if mode == "line":
                marginal.plot(ax=ax, color=cmap(coord_dict[stack_axis]), **kwargs)
            else:
                assert mode == "scatter"
                ax.scatter(
                    *marginal.G.to_arrays(),
                    color=cmap(coord_dict[stack_axis]),
                    **kwargs,
                )
                ax.set_xlabel(marginal.dims[0])

    ax.set_xlabel(label_for_dim(data, ax.get_xlabel()))
    ax.set_ylabel("Spectrum Intensity (arb).")
    ax.set_title(title, fontsize=14)
    ax.set_xlim([other_coord.min().item(), other_coord.max().item()])

    try:
        if inset_ax is not None and not skip_colorbar:
            inset_ax.set_xlabel(stack_axis, fontsize=16)
            fancy_labels(inset_ax)

            cbar(ax=inset_ax, **kwargs)
    except TypeError:
        # already rendered
        pass

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax


@lift_spectrum
def stack_dispersion_plot(
    spectrum: xr.DataArray,
    stack_axis: str = None,
    data_scaling: float = 2.0,
    max_stacks: int = 15,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Generate a stack plot of a 2D spectrum along the specified `stack_axis`.

    Each cut is scaled to fit within the average cut offset multiplied by
    `data_scaling`. Cuts are offset so they intersect the y-axis near their cut value.

    Args
    ----
        spectrum : DataArray
            dataset to plot
        stack_axis : str
            axis to stack along (defaults to the zeroth axis)
        data_scaling : float
            scaling factor for the data
        max_stacks : int
            maximum number of stacks to plot

    Returns
    -------
        fig, ax: figure and axis of the plot
    """
    axes = list(spectrum.dims)
    if len(axes) != 2:
        raise ValueError(f"Spectrum must be 2D, but has dimensions {spectrum.dims}")
    stack_axis = axes[0] if stack_axis is None else stack_axis
    try:
        axes.remove(stack_axis)
    except ValueError:
        raise ValueError(
            f"stack_axis, {stack_axis}, is not one of the data's axes: {axes}"
        )
    plot_axis = axes[0]

    spectrum = spectrum.sortby(stack_axis)
    if max_stacks < len(spectrum.coords[stack_axis]):
        stack_data = rebin(
            spectrum,
            reduction={stack_axis: ceil(len(spectrum[stack_axis]) / max_stacks)},
        )
    else:
        stack_data = spectrum.copy(deep=True)

    cut_values = stack_data.coords[stack_axis].values
    x = stack_data.coords[plot_axis].values

    def normalize_by_max(y: np.ndarray, max: float):
        return (y - y.min()) / max

    raw_ys: list[np.ndarray] = [
        y_values for y_values in stack_data.transpose(stack_axis, ...).values
    ]
    max_height = max([y.max() - y.min() for y in raw_ys])
    normalized_ys = [normalize_by_max(y, max_height) for y in raw_ys]

    average_cut_offset = np.mean(np.diff(cut_values))
    average_y_intercept = np.mean([y[0] for y in normalized_ys])
    scaled_ys = np.array(
        [
            ((y - average_y_intercept) * data_scaling * average_cut_offset) + cut_value
            for y, cut_value in zip(normalized_ys, cut_values)
        ]
    )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.plot(x, scaled_ys.transpose(), **kwargs)

    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel(plot_axis)
    ax.set_ylabel(stack_axis)

    return fig, ax


@save_plot_provenance
def overlapped_stack_dispersion_plot(
    data: DataType,
    stack_axis=None,
    ax=None,
    title=None,
    out=None,
    max_stacks=100,
    use_constant_correction=False,
    transpose=False,
    negate=False,
    s=1,
    scale_factor=None,
    linewidth=1,
    palette=None,
    **kwargs,
):
    data = normalize_to_spectrum(data)

    if stack_axis is None:
        stack_axis = data.dims[0]

    other_axes = list(data.dims)
    other_axes.remove(stack_axis)
    other_axis = other_axes[0]

    stack_coord = data.coords[stack_axis]
    if len(stack_coord.values) > max_stacks:
        data = rebin(
            data,
            reduction=dict(
                [[stack_axis, int(np.ceil(len(stack_coord.values) / max_stacks))]]
            ),
        )

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    if title is None:
        title = "{} Stack".format(data.S.label.replace("_", " "))

    max_over_stacks = np.max(data.values)

    cvalues = data.coords[other_axis].values
    if scale_factor is None:
        maximum_deviation = -np.inf

        for _, marginal in data.G.iterate_axis(stack_axis):
            marginal_values = -marginal.values if negate else marginal.values
            marginal_offset, right_marginal_offset = (
                marginal_values[0],
                marginal_values[-1],
            )

            if use_constant_correction:
                true_ys = marginal_values - marginal_offset
            else:
                true_ys = marginal_values - np.linspace(
                    marginal_offset, right_marginal_offset, len(marginal_values)
                )

            maximum_deviation = np.max([maximum_deviation] + list(np.abs(true_ys)))

        scale_factor = 0.02 * (np.max(cvalues) - np.min(cvalues)) / maximum_deviation

    iteration_order = -1  # might need to fiddle with this in certain cases
    for coord_dict, marginal in list(data.G.iterate_axis(stack_axis))[
        ::iteration_order
    ]:
        coord_value = coord_dict[stack_axis]

        xs = cvalues
        marginal_values = -marginal.values if negate else marginal.values
        marginal_offset, right_marginal_offset = marginal_values[0], marginal_values[-1]

        if use_constant_correction:
            true_ys = (marginal_values - marginal_offset) / max_over_stacks
            ys = scale_factor * true_ys + coord_value
        else:
            true_ys = (
                marginal_values
                - np.linspace(
                    marginal_offset, right_marginal_offset, len(marginal_values)
                )
            ) / max_over_stacks
            ys = scale_factor * true_ys + coord_value

        raw_colors = "black"
        if palette:
            if isinstance(palette, str):
                palette = cm.get_cmap(palette)
            raw_colors = palette(np.abs(true_ys / max_over_stacks))

        if transpose:
            xs, ys = ys, xs

        if isinstance(raw_colors, str):
            plt.plot(xs, ys, linewidth=linewidth, color=raw_colors, **kwargs)
        else:
            plt.scatter(xs, ys, color=raw_colors, s=s, **kwargs)

    x_label = other_axis
    y_label = stack_axis

    if transpose:
        x_label, y_label = y_label, x_label

    ax.set_xlabel(label_for_dim(data, x_label))
    ax.set_ylabel(label_for_dim(data, y_label))

    ax.set_title(title)

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    plt.show()

    return fig, ax
