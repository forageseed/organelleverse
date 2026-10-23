"""Load this checkout when callers explicitly put its src directory on sys.path.

The distributable package is src/__init__.py, mapped by setuptools. This entry
preserves source-only Python and subprocess invocations without a nested folder.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_directory = Path(__file__).resolve().parent
_spec = spec_from_file_location(
    __name__, _directory / "__init__.py",
    submodule_search_locations=[str(_directory)],
)
_module = module_from_spec(_spec)
sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
