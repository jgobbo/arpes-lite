import weakref
from math import floor
from xarray import Dataset
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QGridLayout

from arpes.typing import DataType
from arpes.utilities.qt.data_array_image_view import DataArrayImageView, DataArrayPlot
from arpes.utilities.qt.utils import ReactivePlotRecord, PlotOrientation
from .base import ImageTool


class ChildWindow(QWidget):
    def __init__(self, data: DataType):
        super().__init__()
        self.setLayout(QGridLayout())
        self.spectrum = data.S.spectrum if isinstance(data, Dataset) else data


class ParentTool(ImageTool):
    """
    ImageTool with child windows displaying data with coordinate slices taken from the
    parent tool.
    """

    TITLE = "Parent Tool"

    children: list[tuple[ChildWindow, tuple[str, ...]]]
    child_views: list[ReactivePlotRecord]

    def __init__(self):
        super().__init__()
        self.children = []
        self.child_views = []

    def add_child(self, data: DataType, connected_axes: str | tuple[str, ...]):
        child = ChildWindow(data)
        connected_axes = (
            (connected_axes,) if isinstance(connected_axes, str) else connected_axes
        )
        self.children.append((child, connected_axes))

    def generate_child_marginal(
        self,
        child: ChildWindow,
        connected_axes: tuple[str, ...],
        name: str = None,
    ):
        """
        Generates a marginal plot for this applications's data after selecting along
        `dimensions`. This is used to generate the many different views of a volume in
        the browsable tools.
        """
        layout = child.layout()
        spectrum = child.spectrum

        remaining_dims = [dim for dim in spectrum.dims if dim not in connected_axes]

        if len(remaining_dims) == 1:
            widget = DataArrayPlot(
                name=name,
                root=weakref.ref(child),
                orientation=PlotOrientation.Horizontal,
            )
            self.views[name] = widget
            widget.setMaximumHeight(200)
        else:
            assert len(remaining_dims) == 2
            widget = DataArrayImageView(name=name, root=weakref.ref(child))
            widget.view.setAspectLocked(False)
            self.views[name] = widget

            # TODO: set this with the correct ninety_eight_percentile
            # widget.setHistogramRange(0, self.ninety_eight_percentile)
            widget.setLevels(0.05, 0.95)

        self.child_views.append(
            ReactivePlotRecord(
                dims=connected_axes, view=widget, orientation=PlotOrientation.Horizontal
            )
        )
        layout.addWidget(widget, 0, 0)
        return widget

    def update_reactive_views(
        self,
        coord_slices: dict[str, slice] = None,
        changed_dimensions: set[str] = None,
        force=False,
        keep_levels=True,
    ):
        return super().update_reactive_views(
            coord_slices,
            changed_dimensions,
            force,
            keep_levels,
            self.reactive_views + self.child_views,
        )

    def before_show(self):
        for i, (child, connected_axes) in enumerate(self.children):
            self.generate_child_marginal(
                child,
                connected_axes,
                name=f"child_{i}",
            )

        super().before_show()

    def after_show(self):
        TITLE_BAR_HEIGHT = 32
        TASKBAR_HEIGHT = 50
        screen_size = self.primaryScreen().size()
        screen_width = screen_size.width()
        screen_height = screen_size.height() - TASKBAR_HEIGHT
        parent_proportion = 3 / 5

        self.window.setGeometry(
            0,
            TITLE_BAR_HEIGHT,
            floor(screen_width * parent_proportion),
            screen_height - TITLE_BAR_HEIGHT,
        )

        n_axes = {
            child: (len(child.spectrum.dims) - len(connected_axes))
            for child, connected_axes in self.children
        }
        total_child_axes = sum(n_axes.values())
        assert total_child_axes > 0

        starting_height = TITLE_BAR_HEIGHT
        for child_window, n_child_axes in n_axes.items():
            child_window.show()
            child_height = screen_height // total_child_axes * n_child_axes
            child_window.setGeometry(
                floor(screen_width * parent_proportion),
                starting_height,
                floor(screen_width * (1 - parent_proportion)),
                child_height - TITLE_BAR_HEIGHT,
            )
            starting_height += child_height

        return super().after_show()

    def close(self):
        super().close()
        for child, _ in self.children:
            child.close()
