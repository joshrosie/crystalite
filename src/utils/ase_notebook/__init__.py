"""Copyright (c) Meta Platforms, Inc. and affiliates."""

"""Module for visualizing atomic configurations."""
from src.utils.ase_notebook.backend.svg import (
    concatenate_svgs,
    svg_to_pdf,
)  # noqa: F401
from src.utils.ase_notebook.color import Color  # noqa: F401
from src.utils.ase_notebook.configuration import ViewConfig  # noqa: F401
from src.utils.ase_notebook.data import get_example_atoms  # noqa: F401
from src.utils.ase_notebook.viewer import AseView  # noqa: F401

__version__ = "0.3.2"
