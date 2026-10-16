"""The data passed between our two nodes."""

from typing import TypedDict


class AppState(TypedDict, total=False):
    request: str
    html: str
    file_path: str
