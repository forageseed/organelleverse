"""The public package name must survive the flat on-disk source layout."""
import json
import os
from pathlib import Path
import subprocess
import sys

from tests._paths import PROJECT_ROOT


def test_source_import_outside_checkout_keeps_package_and_resource_paths(tmp_path):
    environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"))
    process = subprocess.run(
        [sys.executable, "-c", (
            "import json, importlib.resources as r; import organelleverse as ov; "
            "from organelleverse._layout import import_root; "
            "from organelleverse.core import OrganelleResult; "
            "print(json.dumps({'file': ov.__file__, 'root': str(import_root()), "
            "'resources': (r.files('organelleverse') / 'annotation' / 'data').is_dir()}))"
        )], cwd=tmp_path, env=environment, capture_output=True, text=True, check=True,
    )
    observed = json.loads(process.stdout)
    assert Path(observed["file"]).resolve() == (PROJECT_ROOT / "src/__init__.py").resolve()
    assert Path(observed["root"]).resolve() == (PROJECT_ROOT / "src").resolve()
    assert observed["resources"]
