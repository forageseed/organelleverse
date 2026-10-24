"""Paths for the explicitly supported source checkout and installed package."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def import_root() -> Path:
    """Pin child interpreters to this package, including a flat source checkout."""
    return PACKAGE_ROOT if PACKAGE_ROOT.name == "src" else PACKAGE_ROOT.parent


def checkout_root() -> Path:
    """The source checkout used by development-only tool discovery."""
    return PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "src" else PACKAGE_ROOT.parent.parent
