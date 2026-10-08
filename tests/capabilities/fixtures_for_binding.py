"""A trivial dataclass used only to exercise the ``dataclass`` parameter codec."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Rectangle:
    width: float
    height: float
