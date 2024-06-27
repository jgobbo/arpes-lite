"""Provides a Qt based implementation of Igor's ImageTool."""

from .base import ImageTool
from .parent_tool import ParentTool
from arpes.typing import DataType
from arpes.utilities.qt import run_tool_in_daemon_process


def _image_tool(data: DataType, **kwargs):
    """Starts the image_tool using an input spectrum."""

    tool = ImageTool()
    tool.set_data(data)
    tool.start(**kwargs)


def _parent_tool(data: DataType, children: list[tuple[DataType, str]], **kwargs):
    """Starts the parent_tool using an input spectrum."""

    tool = ParentTool()
    tool.set_data(data)
    for child in children:
        tool.add_child(*child)
    tool.start(**kwargs)


image_tool = run_tool_in_daemon_process(_image_tool)
parent_tool = run_tool_in_daemon_process(_parent_tool)
