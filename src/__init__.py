"""Early At-Risk Student Detection — reusable library.

This package extracts the core logic from the notebooks so it can be reused by
the CLI (`predict.py`), the analysis scripts in `scripts/`, and the Streamlit
intervention simulator in `app/`.
"""

from . import config

__all__ = ["config"]
