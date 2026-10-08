"""Concrete compute providers.

This package exports no concrete provider and imports no optional dependency.
Providers register through the ``organelleverse.compute_providers`` entry-point
group and are loaded lazily, only after explicit trust, by the
:mod:`organelleverse.compute` discovery/trust layer.
"""
