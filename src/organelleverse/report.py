"""Short reporting API: compute a report, then write it.

``ov.report.build(...)`` constructs a compute-only :class:`OrganelleReport`
(no file); ``ov.report.write(report, output=...)`` materializes it.
"""

from __future__ import annotations

from .reporting import build, write

__all__ = ["build", "write"]
