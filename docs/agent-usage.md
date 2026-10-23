# Install OV and let an agent read its usage guide

Use the project's chosen Python environment (Python 3.11 or newer). Ordinary
installation from the guide branch is:

```bash
uv pip install --python /path/to/project/python "organelleverse @ https://github.com/forageseed/organelleverse/archive/refs/heads/agent/installed-usage-guide.tar.gz"
```

For a reproducible installation, use `https://github.com/forageseed/organelleverse/archive/<reviewed-commit-sha>.tar.gz`. The archive source does not require a Git executable.
Use `.venv/bin/python` on Linux/macOS or `.venv/Scripts/python.exe` on Windows if
that is the project's selected environment. Create a virtual environment with
`uv venv .venv` only if the project does not already have one. The install includes
OV's declared Python dependencies, not every external scientific tool or model.

In that same environment, read the installed instructions:

```bash
python -m organelleverse.agent_guide
python -m organelleverse.agent_guide --skill
```

The first prints the complete usage guide. The second prints a small skill
pointing to that guide. `organelleverse-agent` is an equivalent console entry.
Python users can call `read_guide()` and `read_skill()` from
`organelleverse.agent_guide`.

Save the skill output in the user's project as
`.agents/skills/organelleverse/SKILL.md` if the agent supports that location;
otherwise use its documented skill location or read the guide directly. Preserve
existing skills and project instructions. Refresh the host's catalog or start a
new session if it discovers skills only at startup. Installing a Python package
does not silently register instructions in an arbitrary agent host.

For a host that reads a project `AGENTS.md`, append a short pointer without
replacing existing rules:

> When using OrganelleVerse, first read `python -m organelleverse.agent_guide`
> in the project's selected Python environment. Inspect installed operations and
> signatures for unfamiliar tasks, then check result status and limitations.

## Message to paste into an agent chat

This message uses the maintained installation branch. Replace its archive URL
with a reviewed commit archive when sharing a fixed build.

> 请在当前项目的 Python 3.11+ 环境里，用 uv 普通安装
> `organelleverse @ https://github.com/forageseed/organelleverse/archive/refs/heads/agent/installed-usage-guide.tar.gz`。
> 安装后在同一环境运行 `python -m organelleverse.agent_guide` 并读完指南，按你支持的方式启用包内 skill，保留已有配置。
> 之后根据我的科研需求按需选用 OV，遵守项目的计算和下载规则；安装或检查失败时说明原因。

The receiving agent should use the installed guide to discover functions and
inspect their help and contracts as needed. Public scientific Python calls and
registered calls share validation; native injection interfaces remain available.
An installed package alone cannot make every host automatically read its skill.

## Maintainer verification

`scripts/verify_agent_install.py` builds a wheel from the formal source
and installs it with uv into a fresh environment. Supply `--uv`, `--work-dir` and
`--offline` when using existing cached dependencies. It checks actual installed
module paths, absence of an editable install, resources, module and console
entries, and a project skill export. It does not validate scientific workflows
or demonstrate a particular agent host's automatic activation.
