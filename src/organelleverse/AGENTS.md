# Using the installed OrganelleVerse Python package

This is a usage reference for humans and agents. It does not replace the user's
instructions or the host's permissions. It describes the installed OV version;
installation alone does not prove that an external tool or scientific workflow
is ready.

## Read this guide in the environment that will run the analysis

```bash
python -m organelleverse.agent_guide
```

The installed console command `organelleverse-agent` prints the same guide. Python
callers can use `from organelleverse.agent_guide import read_guide, read_skill`.
`read_guide()` returns this text; `read_skill()` returns the small optional skill.
Neither reader enumerates operations, starts a job, installs dependencies, nor
changes host settings.

`python -m organelleverse.agent_guide --skill` prints the bundled `SKILL.md`.
For agents supporting project skills, save that text as
`.agents/skills/organelleverse/SKILL.md` in the user's project, or use the host's
documented skill location. Preserve an existing skill; review differences before
updating it. A Python install does not make every agent scan package resources.
If skills are unavailable, read this guide directly. A host may need to refresh
its skill catalog or begin a new session after a skill is added.

## Find the operation, then inspect the real API

Use familiar public Python functions directly. When the function or method is
unclear, discover candidates instead of guessing names or loading every schema:

```python
import inspect
import organelleverse as ov

spec = ov.operations.describe("composition.gc_content")
print(spec.description)
print(ov.operations.parameter_schema(spec.operation_id))
print(ov.operations.check_dependencies(spec.operation_id))
print(inspect.signature(ov.composition.gc_content))
print(inspect.getdoc(ov.composition.gc_content))
```

For an unfamiliar task, inspect `ov.operations.list()` to discover the installed
catalog. Versions providing `ov.operations.search` can rank a short candidate list
with `ov.operations.search("GC content", limit=3)`. Check that an optional discovery
entry exists before using it; older releases do not expose every newer helper.
Search ranks candidates; it does not establish scientific suitability. Match the
input type, sequencing technology, organelle, biological question and method
assumptions. Prefer complete public analyses; internal helpers and pipeline steps
are not interchangeable with them. Check alternatives when candidates are
ambiguous. Do not assume a registered operation ID is also a Python function name.
Some JSON operations wrap a differently named Python function. Inspect the actual
module and signature. If the installed version exposes
`ov.operations.public_function(operation_id)`, it can supply that mapping.

For JSON clients, `ov.operations.invocation_schema(operation_id)` describes the
complete invocation envelope. `ov.operations.invoke(operation_id, input=...,
parameters=...)` uses that contract. Python calls need no MCP connection. Hosts
may provide their own approved SDK or execution interface; follow that interface
when compute selection, permissions or durable job tracking require it.

## Read, compute, inspect, and write

Inspect each reader for its accepted formats and metadata; do not infer
scientific roles from filenames alone. Newer versions provide module-owned
readers and `ov.read(path, module=...)`; inspect the installed reader's signature
before using that argument. Some public functions accept a path directly.

This small example uses an existing FASTA file, not a download:

```python
from pathlib import Path
import organelleverse as ov

result = ov.composition.gc_content("genome.fasta", window_size=500)
result.raise_for_failure()
print(result.status, result.summary())
print(result.metrics)
ov.write(result, Path("results/gc"))
```

`ov.write` is the explicit publication boundary. Operations requiring working
files use a managed run store. Inspect each function's return type: some public
functions return plans, deferred plots, paths or ordinary values rather than an
`OrganelleResult`. Do not claim execution merely because a plan was returned.

For `OrganelleResult`, inspect `status`, `errors`, `flags`, `findings`, `artifacts`
and `provenance`. A completed process is not proof of an accepted scientific
result. Report the actual backend and limitations; preserve warnings. An assembly
represented as a circle does not establish a unique physical master circle.
Use declared outputs as inputs to the next operation. Do not invent metrics or
edit a displayed result to make it fit the next call. `suggested_operations` and
the optional `ov.operations.agent_advice(result)` (when exposed by the installed
version) describe compatible next steps, not an independent scientific validation
or permission grant.

## Dependencies and execution location

The Python distribution does not bundle every assembler, executable, database or
model. Inspect dependencies before running. A dependency report is a preflight
check, not proof of a successful real-tool run. Report missing requirements; do
not silently replace a failed method or invent an unavailable backend.

OV supplies scientific functions and contracts. The caller or host supplies
execution targets, credentials, approvals and remote job lifecycle. Ordinary
Python executes where its interpreter runs. Selecting a server in a conversation
does not itself move a Python call there: use that server's interpreter or the
host's compute interface. Respect the user's download and compute policies.
Reconnect by querying an existing remote job handle instead of resubmitting it.

When sharing results, record the installed OV version, input identifiers,
parameters, actual tool versions, output paths and material limitations. Report
package installation, dependency readiness, execution and scientific acceptance
as distinct observations.
