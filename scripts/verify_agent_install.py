"""Build formal OV source and check its guide after an ordinary uv wheel install.

Run with the formal project's build interpreter. --work-dir holds all build,
environment and receipt files; --offline forbids dependency downloads.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"}
    # An inherited PYTHONPATH must not make the installed-package check see source.
    env.pop("PYTHONPATH", None)

    def run(name: str, command: list[str], cwd: Path = root) -> str:
        result = subprocess.run(
            command, cwd=cwd, env=env, text=True, encoding="utf-8",
            errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=600,
        )
        (work / f"{name}.log").write_text(result.stdout, encoding="utf-8")
        print(f"{name}: exit {result.returncode}", flush=True)
        if result.returncode:
            print(result.stdout[-5000:], flush=True)
            raise SystemExit(result.returncode)
        return result.stdout

    import organelleverse
    from organelleverse import agent_guide

    source = {
        "python": sys.executable,
        "package": organelleverse.__file__,
        "guide_module": agent_guide.__file__,
    }
    assert Path(agent_guide.__file__).resolve().is_relative_to(root / "src"), source
    (work / "formal-source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(json.dumps(source), flush=True)

    backend = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "build-system"
    ]["build-backend"]
    if backend == "setuptools.build_meta":
        run("build", [
            sys.executable, "setup.py", "build", "--build-base", str(work / "build"),
            "bdist_wheel", "--dist-dir", str(work / "dist"), "--bdist-dir", str(work / "bdist"),
        ])
    else:
        run("build", [
            sys.executable, "-m", "build", "--wheel", "--no-isolation",
            "--outdir", str(work / "dist"),
        ])
    wheels = list((work / "dist").glob("organelleverse-*.whl"))
    assert len(wheels) == 1, wheels
    wheel = wheels[0]
    uv = str(args.uv.resolve())
    offline = ["--offline"] if args.offline else []
    run("venv", [uv, *offline, "venv", "--python", sys.executable, str(work / "venv")])
    scripts = work / "venv" / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    run("install", [uv, *offline, "pip", "install", "--python", str(python), str(wheel)])
    outside = work / "consumer-project"
    outside.mkdir(exist_ok=True)
    probe = run("installed-probe", [str(python), "-I", "-c", """
import importlib.metadata as md
import json
from pathlib import Path
import sys
import organelleverse as ov
from organelleverse import agent_guide
from importlib.resources import files
dist = md.distribution('organelleverse')
url = json.loads(dist.read_text('direct_url.json') or '{}')
assert not url.get('dir_info', {}).get('editable', False), url
assert Path(ov.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), ov.__file__
assert Path(agent_guide.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert not list(Path(dist.locate_file('')).glob('__editable__*organelleverse*'))
guide = agent_guide.read_guide()
skill = agent_guide.read_skill()
assert guide == files('organelleverse').joinpath('AGENTS.md').read_text(encoding='utf-8')
assert skill.startswith('---\\nname: organelleverse\\n')
assert 'organelleverse.operations' not in sys.modules
Path('guide.txt').write_text(guide, encoding='utf-8')
target = Path('.agents/skills/organelleverse/SKILL.md')
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(skill, encoding='utf-8')
print(json.dumps({'python': sys.executable, 'package': ov.__file__,
 'guide_module': agent_guide.__file__, 'version': dist.version,
 'direct_url': url, 'guide_chars': len(guide), 'skill_chars': len(skill),
 'skill_file': str(target.resolve())}, indent=2))
"""], outside)
    guide = (outside / "guide.txt").read_text(encoding="utf-8")
    skill = (outside / ".agents/skills/organelleverse/SKILL.md").read_text(encoding="utf-8")
    console = scripts / ("organelleverse-agent.exe" if os.name == "nt" else "organelleverse-agent")
    for name, command, expected in [
        ("module-guide", [str(python), "-I", "-m", "organelleverse.agent_guide"], guide),
        ("module-skill", [str(python), "-I", "-m", "organelleverse.agent_guide", "--skill"], skill),
        ("console-guide", [str(console)], guide),
        ("console-skill", [str(console), "--skill"], skill),
    ]:
        assert run(name, command, outside) == expected, name
    assert agent_guide.read_guide() == guide
    assert agent_guide.read_skill() == skill
    assert Path(agent_guide.__file__).resolve().is_relative_to(root / "src")
    (work / "receipt.json").write_text(json.dumps({
        "formal_source": source, "wheel": str(wheel), "installed": json.loads(probe),
        "offline": args.offline, "checks": [
            "noneditable_uv_install", "installed_module_origin", "no_editable_hook",
            "package_resources", "project_skill_written", "no_catalog_loading",
            "module_guide", "module_skill", "console_guide", "console_skill",
            "formal_and_installed_resources_equal",
        ],
    }, indent=2), encoding="utf-8")
    print(f"PASS: {work / 'receipt.json'}", flush=True)


if __name__ == "__main__":
    main()
