from arpes.utilities.qt.utils import PlotOrientation
from PyQt5 import QtGui, QtWidgets
import numpy as np
import xarray as xr
import weakref
from typing import override
from collections import defaultdict


from arpes.utilities.qt.ui import (
    KeyBinding,
    Key,
    KeyboardModifier,
    horizontal,
    vertical,
    tabs,
)
from arpes.utilities.qt import (
    qt_info,
    BasicHelpDialog,
    SimpleWindow,
    SimpleApp,
)

from .AxisInfoWidget import AxisInfoWidget
from .BinningInfoWidget import BinningInfoWidget

__all__ = (
    "ImageTool",
    "image_tool",
)

qt_info.setup_pyqtgraph()


class ImageToolWindow(SimpleWindow):
    """
    The application window for `QtTool`.

    QtToolWindow was the first Qt-Based Tool that I built for PyARPES. Much of its
    structure was ported to SimpleWindow and borrowed ideas from when I wrote DAQuiri.
    As a result, the structure is essentially now to define just the handlers and any
    lifecycle hooks (close, etc.)
    """

    HELP_DIALOG_CLS = BasicHelpDialog
    SCROLL_INCREMENT = 5
    INCREMENT_MULTIPLIER = 5

    def compile_key_bindings(self):
        return super().compile_key_bindings() + [  # already includes Help and Close
            KeyBinding(
                "Scroll Cursor",
                [
                    Key.Key_Left,
                    Key.Key_Right,
                    Key.Key_Up,
                    Key.Key_Down,
                ],
                self.scroll,
            ),
            KeyBinding(
                "Reset Intensity",
                [Key.Key_I],
                self.reset_intensity,
            ),
            KeyBinding(
                "Scroll Z-Cursor",
                [
                    Key.Key_N,
                    Key.Key_M,
                ],
                self.scroll_z,
            ),
            KeyBinding(
                "Center Cursor",
                [Key.Key_C],
                self.center_cursor,
            ),
            KeyBinding(
                "Transpose - Roll Axis",
                [Key.Key_Y],
                self.transpose_roll,
            ),
            KeyBinding(
                "Transpose - Swap Front Axes",
                [Key.Key_T],
                self.transpose_swap,
            ),
        ]

    def center_cursor(self, event):
        self.app().center_cursor()

    def transpose_roll(self, event):
        self.app().transpose_to_front(-1)

    def transpose_swap(self, event):
        self.app().transpose_to_front(1)

    def reset_intensity(self, event: QtGui.QKeyEvent):
        self.app().reset_intensity()

    def get_increment(self, event: QtGui.QKeyEvent):
        if event.modifiers() & KeyboardModifier.ShiftModifier:
            return self.SCROLL_INCREMENT * self.INCREMENT_MULTIPLIER
        elif event.modifiers() & KeyboardModifier.ControlModifier:
            return self.SCROLL_INCREMENT // self.INCREMENT_MULTIPLIER
        return self.SCROLL_INCREMENT

    def scroll_z(self, event: QtGui.QKeyEvent):
        key_map = {
            Key.Key_N: (2, -1),
            Key.Key_M: (2, 1),
        }

        dim_i, delta = key_map.get(event.key())
        delta *= self.get_increment(event)
        if delta is not None and self.app() is not None:
            self.app().scroll(dim_i, delta)

    def scroll(self, event: QtGui.QKeyEvent):
        key_map = {
            Key.Key_Left: (0, -1),
            Key.Key_Right: (0, 1),
            Key.Key_Down: (1, -1),
            Key.Key_Up: (1, 1),
        }

        dim_i, delta = key_map.get(event.key())
        delta *= self.get_increment(event)
        if delta is not None and self.app() is not None:
            self.app().scroll(dim_i, delta)


class ImageTool(SimpleApp):
    """QtTool is an implementation of Image/Bokeh Tool based on PyQtGraph and PyQt5.

    For now we retain a number of the metaphors from BokehTool, including a "context"
    that stores the state, and can be used to programmatically interface with the tool.
    """

    TITLE = "Image Tool"
    WINDOW_CLS = ImageToolWindow
    WINDOW_SIZE = (12, 10)

    def __init__(self):
        """Initialize attributes to safe empty values."""
        super().__init__()
        self.data = None

        self.content_layout = None
        self.main_layout = None

        self.axis_info_widgets = []
        self.binning_info_widgets = []
        self.kspace_info_widgets = []

        self._binning = None

    @override
    def update_cursor_position(self, new_cursor, **kwargs):
        super().update_cursor_position(new_cursor, **kwargs)

        for widget in self.axis_info_widgets + self.binning_info_widgets:
            widget.recompute()

    def center_cursor(self):
        """Scrolls so that the cursors are in the center of the data volume."""
        centers = {dim: self.spectrum.sizes[dim] / 2 for dim in self.spectrum.dims}
        new_cursor = {
            dim: (center - self._binning[dim] / 2, center + self._binning[dim] / 2)
            for dim, center in centers.items()
        }

        for dim, cursors in self.registered_cursors.items():
            for cursor in cursors:
                cursor.setRegion(new_cursor[dim])

    def scroll(self, dim_i: int, delta: int):
        """Scroll the cursor with dim specified by dim_i by delta."""
        dim = self.spectrum.dims[dim_i]
        cursor = self.context["cursor"].copy()
        new_cursor_position = tuple(
            old_position + delta for old_position in cursor[dim]
        )

        def clamp(positions: tuple[float, float], dim: str):
            start, end = positions
            if start < 0:
                start = 0
                end = self._binning[dim]
            elif end > self.spectrum.sizes[dim]:
                end = self.spectrum.sizes[dim]
                start = end - self._binning[dim]
            return start, end

        cursor[dim] = clamp(new_cursor_position, dim)
        for dim_cursor in self.registered_cursors[dim]:
            dim_cursor.setRegion(cursor[dim])

    @property
    def binning(self):
        """The binning on each axis in number of indices."""
        if self._binning is None:
            return {dim: 1 for dim in self.spectrum.dims}

        return self._binning

    @binning.setter
    def binning(self, values: dict[str, int]):
        """Set the desired axis binning."""
        different_binnings = {
            dim: n_bins
            for dim, n_bins in values.items()
            if n_bins != self._binning[dim]
        }
        self._binning = values

        for dim in different_binnings:
            cursors = self.registered_cursors[dim]
            for cursor in cursors:
                cursor.set_width(self._binning[dim])

    def change_cursor_dims(self, dim_mapping: dict[str, str]) -> None:
        """Change the dimensions of the cursors according to the provided mapping."""
        old_cursors = self.registered_cursors
        self.registered_cursors = defaultdict(list)
        for old_dim, cursor_list in old_cursors.items():
            new_dim = dim_mapping[old_dim]
            new_bounds = [0, self.spectrum.sizes[new_dim]]
            new_width = self.binning[new_dim]
            for cursor in cursor_list:
                cursor.setBounds(new_bounds)
                cursor.set_width(new_width)
                self.reconnect_cursor(new_dim, cursor)

    def transpose(self, transpose_order: list[str]):
        """Transpose dimensions into the specified order and redraw."""
        original_order = list(self.spectrum.dims)
        dim_mapping = {
            original_dim: new_dim
            for original_dim, new_dim in zip(original_order, transpose_order)
        }
        self.spectrum = self.spectrum.transpose(*transpose_order, ...)

        for reactive_view in self.reactive_views:
            old_dims = reactive_view.dims
            reactive_view.dims = tuple(dim_mapping[dim] for dim in old_dims)

        for widget in self.axis_info_widgets + self.binning_info_widgets:
            widget.recompute()

        self.change_cursor_dims(dim_mapping)
        cursor_positions = self.context["cursor"]
        for dim, cursors in self.registered_cursors.items():
            for cursor in cursors:
                cursor.setRegion(cursor_positions[dim])

        self.update_reactive_views(force=True, keep_levels=False)

    def transpose_to_front(self, dim: str | int):
        """Transpose dim to the front so that it is in the main marginal."""
        dim = dim if isinstance(dim, str) else self.spectrum.dims[dim]

        order = list(self.spectrum.dims)
        order.remove(dim)
        order = [dim] + order
        self.transpose(order)

    def configure_image_widgets(self):
        """
        Configure array marginals for the input data.

        Depending on the array dimensionality, we need a different number and variety
        of marginals. This is as easy as specifying which marginals we select over and
        handling the rest dynamically.

        An additional complexity is that we also handle the cursor registration here.
        """
        dims = self.spectrum.dims
        if len(dims) == 2:
            self.generate_marginal_for(
                (), 1, 0, "xy", cursors=True, layout=self.content_layout
            )
            self.generate_marginal_for(
                (dims[1],),
                0,
                0,
                "x",
                orientation=PlotOrientation.Horizontal,
                layout=self.content_layout,
            )
            self.generate_marginal_for(
                (dims[0],),
                1,
                1,
                "y",
                orientation=PlotOrientation.Vertical,
                layout=self.content_layout,
            )

            self.views["xy"].view.setYLink(self.views["y"])
            self.views["xy"].view.setXLink(self.views["x"])

        if len(dims) == 3:
            self.generate_marginal_for(
                (dims[1], dims[2]),
                0,
                0,
                "x",
                orientation=PlotOrientation.Horizontal,
                layout=self.content_layout,
            )
            self.generate_marginal_for(
                (dims[1],), 1, 0, "xz", cursors=True, layout=self.content_layout
            )
            self.generate_marginal_for(
                (dims[2],), 2, 0, "xy", cursors=True, layout=self.content_layout
            )
            self.generate_marginal_for(
                (dims[0], dims[1]),
                1,
                1,
                "z",
                orientation=PlotOrientation.Horizontal,
                layout=self.content_layout,
            )
            self.generate_marginal_for(
                (dims[0], dims[2]),
                2,
                2,
                "y",
                orientation=PlotOrientation.Vertical,
                layout=self.content_layout,
            )
            self.generate_marginal_for(
                (dims[0],), 2, 1, "yz", cursors=True, layout=self.content_layout
            )

            self.views["xy"].view.setYLink(self.views["y"])
            self.views["xy"].view.setXLink(self.views["x"])
            self.views["xz"].view.setXLink(self.views["z"])
            self.views["xz"].view.setXLink(self.views["xy"].view)

        if len(dims) == 4:
            self.generate_marginal_for(
                (dims[1], dims[3]), 0, 0, "xz", layout=self.content_layout
            )
            self.generate_marginal_for(
                (dims[2], dims[3]), 1, 0, "xy", cursors=True, layout=self.content_layout
            )
            self.generate_marginal_for(
                (dims[0], dims[2]), 1, 1, "yz", layout=self.content_layout
            )
            self.generate_marginal_for(
                (dims[0], dims[1]), 0, 1, "zw", cursors=True, layout=self.content_layout
            )

    def construct_axes_tab(self):
        """Controls for axis order and transposition."""
        inner_items = [
            AxisInfoWidget(axis_name=axis_name, root=weakref.ref(self))
            for axis_name in self.spectrum.dims
        ]
        return horizontal(*inner_items), inner_items

    def construct_binning_tab(self):
        """This tab controls the degree of binning around the cursor."""
        inner_items = [
            BinningInfoWidget(axis_name=dim, root=weakref.ref(self))
            for dim in self.spectrum.dims
        ]

        return horizontal(*inner_items), inner_items

    # TODO
    def construct_color_tab(self): ...

    def construct_help_tab(self):
        """The help tab."""
        texts = [
            "Controls:",
            "   Move main cursor - ↓ ↑ ← →, Shift for 5x, Ctrl for 1/5x",
            "   Move Z-cursor - n/m, Shift for 5x, Ctrl for 1/5x",
            "   Reset intensity - i    Center cursor - c",
            "   Transpose Main - t    Swap Main - y",
        ]
        return vertical(*[QtWidgets.QLabel(t) for t in texts])

    def add_contextual_widgets(self):
        """Adds the widgets for the contextual controls at the bottom."""
        axes_tab, self.axis_info_widgets = self.construct_axes_tab()
        binning_tab, self.binning_info_widgets = self.construct_binning_tab()

        self.tabs = tabs(
            ["Binning", binning_tab],
            ["Axes", axes_tab],
            ["Help", self.construct_help_tab()],
        )
        self.tabs.setFixedHeight(qt_info.inches_to_px(2))

        self.main_layout.addLayout(self.content_layout, 0, 0)
        self.main_layout.addWidget(self.tabs, 1, 0)

    def layout(self):
        """Initialize the layout components."""
        self.main_layout = QtWidgets.QGridLayout()
        self.content_layout = QtWidgets.QGridLayout()
        return self.main_layout

    def before_show(self):
        """Lifecycle hook for configuration before app show."""
        self.configure_image_widgets()
        self.add_contextual_widgets()
        from matplotlib import colormaps

        if self.spectrum.min() >= 0.0:
            self.set_colormap(colormaps["magma"])
        else:
            self.set_colormap(colormaps["RdBu_r"])

    def after_show(self):
        """Initialize application state after app show.

        To do this, we need to set the initial cursor location, and call update
        which forces a rerender.
        """
        # basic state initialization
        self.context.update(
            {
                "cursor": {dim: (0, 1) for dim in self.spectrum.dims},
            }
        )

        # Display the data
        self.center_cursor()
        self.reset_intensity()

    def reset_intensity(self):
        """Autoscales intensity in each marginal plot."""
        self.update_reactive_views(force=True, keep_levels=False)

    def set_data(self, data: xr.Dataset):
        """Sets the current data to a new value and resets binning."""

        if np.any(np.isnan(data)):
            data = data.fillna(0)

        self.data = data
        self._binning = {dim: 1 for dim in self.spectrum.dims}
