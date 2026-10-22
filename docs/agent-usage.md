# Install OV and let an agent read its usage guide

Use the project's chosen Python environment (Python 3.11 or newer). Ordinary
installation from the guide branch is:

```bash
uv pip install --python /path/to/project/python "organelleverse @ git+https://github.com/forageseed/organelleverse.git@agent/installed-usage-guide"
```

For a reproducible installation, replace the branch with a reviewed commit SHA.
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

Replace the branch below with a reviewed commit when sharing an exact build.

> 请在当前项目选定的 Python 3.11+ 环境中，用 uv 普通安装
> `organelleverse @ git+https://github.com/forageseed/organelleverse.git@agent/installed-usage-guide`，
> 不要使用 editable 安装。先确认已有环境，未有环境才创建项目 `.venv`，保留现有代码、依赖约束和凭据。
> 安装后使用同一解释器运行 `-m organelleverse.agent_guide` 并读取输出，再运行
> `-m organelleverse.agent_guide --skill`，将输出保存到你支持的项目 skill 目录
> （支持时使用 `.agents/skills/organelleverse/SKILL.md`）；已有同名文件先核对，不直接覆盖。
> 若你读取项目 `AGENTS.md`，在保留原规则的前提下补充一条使用 OV 前读取该指南的提示。
> 核对实际导入路径和 skill 文件；宿主需要时刷新技能目录。之后遇到相关分析任务，按指南和
> 实际 API 自主选择合适功能，检查依赖、结果状态与证据限制，遵守我的计算和下载要求。
> 如果安装或入口检查失败，报告具体原因，不声称已配置成功。

## Maintainer verification

`scripts/verify_agent_install.py` builds a wheel from the formal source
and installs it with uv into a fresh environment. Supply `--uv`, `--work-dir` and
`--offline` when using existing cached dependencies. It checks actual installed
module paths, absence of an editable install, resources, module and console
entries, and a project skill export. It does not validate scientific workflows
or demonstrate a particular agent host's automatic activation.
