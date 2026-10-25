---
name: organelleverse
description: Use the installed OrganelleVerse Python package for plant organelle analyses. Read its version-matched usage guide and discover suitable public operations before unfamiliar calls.
---

# OrganelleVerse

In the Python environment selected for the user's analysis, run
`python -m organelleverse.agent_guide` and read its output. Alternatively:

```python
from organelleverse.agent_guide import read_guide
print(read_guide())
```

Use the guide's discovery and inspection entries for unfamiliar operations, then
call the public Python API. Check method suitability, dependencies, return type,
result status, evidence and limitations. Use the host's execution interface when
the user's compute selection or permissions require it. This skill provides
usage guidance; it grants no additional execution or download permissions.

If that environment cannot import OV, report the missing installation or wrong
interpreter before choosing an installation source. Do not silently switch the
analysis to another environment or invent a function name.
