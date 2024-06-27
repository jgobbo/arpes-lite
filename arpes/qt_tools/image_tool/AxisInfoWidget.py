"""A widget providing rudimentary information about an axis on a DataArray."""

# pylint: disable=import-error

from PyQt5 import QtWidgets
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ImageTool
    from weakref import ReferenceType

__all__ = ("AxisInfoWidget",)


class AxisInfoWidget(QtWidgets.QGroupBox):
    """A widget providing some rudimentary axis information."""

    def __init__(
        self, parent=None, root: "ReferenceType" = None, axis_name: str = None
    ):
        """Configure inner widgets for axis info, and transpose to front button."""
        super().__init__(title=axis_name, parent=parent)

        self.layout = QtWidgets.QGridLayout(self)

        self.label = QtWidgets.QLabel("Cursor: ")
        self.transpose_button = QtWidgets.QPushButton("To Front")
        self.transpose_button.clicked.connect(self.on_transpose)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.transpose_button)

        self.axis_name = axis_name
        self._root = root
        self.setLayout(self.layout)
        self.recompute()

    @property
    def root(self) -> "ImageTool":
        """Unwraps the weakref to the parent application."""
        return self._root()

    def recompute(self):
        """Force a recomputation of dependent UI state: here, the title and text."""
        self.setTitle(self.axis_name)
        try:
            cursor_indices = self.root.context["cursor"][self.axis_name]
            self.label.setText(
                f"Cursor: {cursor_indices[0]:.1f}, {cursor_indices[1]:.1f}"
            )
        except KeyError:
            pass

    def on_transpose(self):
        """This UI control lets you tranpose the axis it refers to to the front."""
        try:
            self.root.transpose_to_front(self.axis_name)
        except Exception:
            pass
