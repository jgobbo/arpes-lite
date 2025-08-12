"""Application infrastructure for apps/tools which browse a data volume."""

import sys
from math import floor
from typing import Any, TYPE_CHECKING
import weakref
import numpy as np
import xarray as xr
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QWidget
from collections import defaultdict

from arpes.utilities.qt.ui import CursorRegion
from arpes.typing import DataType
from arpes.settings import SETTINGS
from .data_array_image_view import DataArrayImageView, DataArrayPlot
from .utils import PlotOrientation, ReactivePlotRecord


if TYPE_CHECKING:
    from .windows import SimpleWindow

__all__ = ["SimpleApp"]


class SimpleApp(QApplication):
    """
    Has all of the layout information and business logic for an interactive data
    browsing utility using PyQt5.
    """

    WINDOW_CLS = None
    WINDOW_SIZE = (4, 4)
    TITLE = "Untitled Tool"

    _data: xr.Dataset = None
    spectrum: xr.DataArray = None

    def __init__(self):
        """Only interesting thing on init is to make a copy of the user settings."""
        super().__init__(sys.argv)
        self._ninety_eight_percentile = None
        self.settings = None
        self._window = None
        self._layout = None

        self.context = {}

        self.views = {}
        self.reactive_views: list[ReactivePlotRecord] = []
        self.registered_cursors: dict[str, list[CursorRegion]] = defaultdict(list)

        self.settings = SETTINGS.copy()

    def copy_to_clipboard(self, value: Any) -> None:
        """Attempts to copy the value to the clipboard, or else prints."""
        try:
            import pyperclip
            import pprint

            pyperclip.copy(pprint.pformat(value))
        except ImportError:
            pass
        finally:
            import pprint

            print(pprint.pformat(value))

    @property
    def data(self) -> DataType:
        """
        Read data from the cached attribute.

        This is a property as opposed to a plain attribute
        in order to facilitate rendering datasets with several
        data_vars.
        """
        return self._data

    @data.setter
    def data(self, new_data: DataType):
        self._data = new_data
        self.spectrum = (
            new_data.S.spectrum if isinstance(new_data, xr.Dataset) else new_data
        )

    def close(self):
        """
        Graceful shutdown. Tell each view to close and drop references so GC happens.
        """
        for v in self.views.values():
            v.close()

        self.views = {}
        self.reactive_views = []

    @property
    def ninety_eight_percentile(self):
        """
        Calculates the 98 percentile of data so colorscale is not outlier dependent.
        """
        if self._ninety_eight_percentile is not None:
            return self._ninety_eight_percentile

        self._ninety_eight_percentile = np.percentile(self.spectrum.values, 98)
        return self._ninety_eight_percentile

    def print(self, *args, **kwargs):
        """Forwards printing to the application so it ends up in Jupyter."""
        self.window.window_print(*args, **kwargs)

    @staticmethod
    def build_pg_cmap(colormap):
        """Converts a matplotlib colormap to one suitable for pyqtgraph.

        pyqtgraph uses its own colormap format but for consistency and aesthetic
        reasons we want to use the ones from matplotlib. This will sample the colors
        from the colormap and convert it into an array suitable for pyqtgraph.
        """
        sampling_array = np.linspace(0, 1, 5)
        sampled_colormap = colormap(sampling_array) * 255

        return pg.ColorMap(
            pos=np.linspace(0, 1, len(sampled_colormap)), color=sampled_colormap
        )

    def set_colormap(self, colormap):
        """Finds all `DataArrayImageView` instances and sets their color palette."""
        import matplotlib.cm

        if isinstance(colormap, str):
            colormap = matplotlib.cm.get_cmap(colormap)

        cmap = self.build_pg_cmap(colormap)
        for view in self.views.values():
            if isinstance(view, DataArrayImageView):
                view.setColorMap(cmap)

    def generate_marginal_for(
        self,
        dimensions: tuple[str],
        row: int,
        column: int,
        name: str = None,
        orientation: PlotOrientation = PlotOrientation.Horizontal,
        cursors: bool = False,
        layout=None,
    ):
        """
        Generates a marginal plot for this applications's data after selecting along
        `dimensions`. This is used to generate the many different views of a volume in
        the browsable tools.
        """
        layout = self._layout if layout is None else layout

        remaining_dims = [dim for dim in self.spectrum.dims if dim not in dimensions]

        if len(remaining_dims) == 1:
            widget = DataArrayPlot(
                name=name, root=weakref.ref(self), orientation=orientation
            )
            self.views[name] = widget

            if orientation == PlotOrientation.Horizontal:
                widget.setMaximumHeight(200)
            else:
                widget.setMaximumWidth(200)

            if cursors:
                remaining_dim = remaining_dims[0]
                cursor = CursorRegion(
                    orientation=(
                        CursorRegion.Horizontal
                        if orientation == PlotOrientation.Vertical
                        else CursorRegion.Vertical
                    ),
                    movable=True,
                    bounds=[
                        0,
                        self.spectrum.sizes[remaining_dim],
                    ],
                )
                widget.addItem(cursor, ignoreBounds=False)
                self.connect_cursor(remaining_dim, cursor)
        else:
            assert len(remaining_dims) == 2
            widget = DataArrayImageView(name=name, root=weakref.ref(self))
            widget.view.setAspectLocked(False)
            self.views[name] = widget

            # TODO: set the bottom level intelligently
            widget.setHistogramRange(0, self.ninety_eight_percentile)
            widget.setLevels(0.05, 0.95)

            if cursors:
                for dim, orientation in zip(
                    remaining_dims, [CursorRegion.Vertical, CursorRegion.Horizontal]
                ):
                    cursor = CursorRegion(
                        orientation=orientation,
                        movable=True,
                        bounds=[0, self.spectrum.sizes[dim]],
                    )
                    widget.addItem(cursor, ignoreBounds=True)
                    self.connect_cursor(dim, cursor)

        self.reactive_views.append(
            ReactivePlotRecord(dims=dimensions, view=widget, orientation=orientation)
        )
        layout.addWidget(widget, row, column)
        return widget

    def connect_cursor(self, dimension: str, the_line: CursorRegion) -> None:
        """Connect a cursor to a line control.

        without weak references we get a circular dependency here
        because `the_line` is owned by a child of `self` but we are
        providing self to a closure which is retained by `the_line`.
        """
        self.registered_cursors[dimension].append(the_line)
        owner = weakref.ref(self)

        def connected_cursor(line: CursorRegion):
            new_cursor = owner().context["cursor"].copy()
            new_cursor[dimension] = line.getRegion()
            owner().update_cursor_position(new_cursor)

        the_line.sigRegionChanged.connect(connected_cursor)

    def reconnect_cursor(self, dimension: str, the_line: CursorRegion) -> None:
        """
        Reconnect a cursor to a line control. User must remove the line from
        registered_cursors before calling this function or there will be a duplicate.
        """
        the_line.sigRegionChanged.disconnect()
        self.connect_cursor(dimension, the_line)

    def update_cursor_position(
        self, new_cursor: dict[str, tuple[float, float]], **kwargs
    ) -> None:
        """
        Sets the current cursor positions. If the cursor has changed, this also
        updates all reactive views.

        Args:
            new_cursor: The new cursor positions.
            kwargs: Additional arguments to pass to `update_reactive_views`.
        """
        old_cursor = self.context["cursor"].copy()
        changed_dimensions = set(
            dim for dim in new_cursor if new_cursor[dim] != old_cursor[dim]
        )
        if not changed_dimensions:
            return
        self.context["cursor"] = new_cursor

        def indices_to_values(indices: float, dim: str):
            coord_values = self.spectrum.coords[dim].values

            # TODO: interpolate for irregularly spaced coords
            def index_to_value(index):
                return coord_values[0] + index * (coord_values[1] - coord_values[0])

            return tuple(index_to_value(index) for index in indices)

        self.context["value_cursor"] = {
            dim: indices_to_values(indices, dim) for dim, indices in new_cursor.items()
        }
        cursor_text = " ".join(
            f"{dim}:({values[0]:.3g}, {values[1]:.3g})"
            for dim, values in self.context["value_cursor"].items()
        )
        self.window.statusBar().showMessage(cursor_text)

        self.update_reactive_views(
            changed_dimensions=changed_dimensions,
            **kwargs,
        )

    def update_reactive_views(
        self,
        coord_slices: dict[str, slice] = None,
        changed_dimensions: set[str] = None,
        force=False,
        keep_levels=True,
        reactive_views=None,
    ):
        """
        Updates all reactive views based on the specified slices.

        Args:
            coord_slices: The slices to apply to the data.
            changed_dimensions: The dimensions that have changed.
            force: If true, will update all views regardless of whether they have
                changed.
            keep_levels: If true, will keep the existing colormap levels.
        """
        if coord_slices is None:
            cursor: dict[str, tuple[float, float]] = self.context["cursor"]
            coord_slices = {dim: slice(*position) for dim, position in cursor.items()}
        changed_dimensions = set() if changed_dimensions is None else changed_dimensions

        def safe_slice(unsafe_slice: slice, dim: str):
            start, stop = floor(unsafe_slice.start), floor(unsafe_slice.stop)
            dim_length = self.spectrum.sizes[dim]
            start, stop = np.clip(start, 0, dim_length), np.clip(stop, 0, dim_length)
            if start == stop:
                stop = start + 1
            start, stop = np.clip(start, 0, dim_length), np.clip(stop, 0, dim_length)
            if start == stop:
                start = stop - 1
            return slice(start, stop)

        safe_slices = {
            dim: safe_slice(unsafe_slice, dim)
            for dim, unsafe_slice in coord_slices.items()
        }
        reactive_views = (
            self.reactive_views if reactive_views is None else reactive_views
        )
        for plot_record in reactive_views:
            if set(plot_record.dims).intersection(changed_dimensions) or force:
                reactive_slices = {dim: safe_slices[dim] for dim in plot_record.dims}
                view = plot_record.view

                if isinstance(view, DataArrayImageView):
                    image_data = view.root.spectrum.isel(**reactive_slices)
                    if reactive_slices:
                        image_data = image_data.mean(list(reactive_slices.keys()))
                    view.setImage(image_data, keep_levels=keep_levels)

                elif isinstance(view, DataArrayPlot):
                    plot_data = view.root.spectrum.isel(**reactive_slices)
                    if reactive_slices:
                        plot_data = plot_data.mean(list(reactive_slices.keys()))

                    cursors = [
                        plot_item
                        for plot_item in view.getPlotItem().items
                        if isinstance(plot_item, CursorRegion)
                    ]
                    view.clear()
                    for cursor in cursors:
                        view.addItem(cursor)

                    view.plot(plot_data)

    def before_show(self):
        """Lifecycle hook."""
        pass

    def after_show(self):
        """Lifecycle hook."""
        pass

    def layout(self):
        """
        Hook for defining the application layout. This needs to be provided by
        subclasses.
        """
        raise NotImplementedError

    @property
    def window(self) -> "SimpleWindow":
        """Gets the window instance on the current application."""
        return self._window

    def start(self):
        """Starts the Qt application, configures the window, and begins Qt execution."""

        from arpes.utilities.qt import qt_info

        qt_info.init_from_app(self)

        self._window = self.WINDOW_CLS()
        self.window.resize(*qt_info.inches_to_px(self.WINDOW_SIZE))
        self.window.setWindowTitle(self.TITLE)

        self.central_widget = QWidget()
        self._layout = self.layout()
        self.central_widget.setLayout(self._layout)
        self.window.setCentralWidget(self.central_widget)
        self.window.app = weakref.ref(self)

        self.before_show()
        self.window.show()
        self.after_show()

        qt_info.apply_settings_to_app(self)

        self.exec()
