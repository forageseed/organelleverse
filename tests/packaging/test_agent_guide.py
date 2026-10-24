"""The installed guide is readable outside a checkout without loading a catalog."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

from organelleverse.agent_guide import read_guide, read_skill


class AgentGuideTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as cwd:
            return subprocess.run(
                [sys.executable, "-I", *args],
                cwd=cwd,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
                timeout=30,
            )

    def test_module_entry_prints_packaged_guide_outside_checkout(self) -> None:
        self.assertEqual(self._run("-m", "organelleverse.agent_guide").stdout, read_guide())

    def test_skill_entry_prints_complete_skill_outside_checkout(self) -> None:
        text = read_skill()
        self.assertTrue(text.startswith("---\nname: organelleverse\n"))
        self.assertEqual(self._run("-m", "organelleverse.agent_guide", "--skill").stdout, text)

    def test_reading_guide_does_not_load_scientific_catalog(self) -> None:
        process = self._run(
            "-c",
            "import json,sys; from organelleverse.agent_guide import read_guide,read_skill; "
            "read_guide(); read_skill(); "
            "print(json.dumps([n for n in sys.modules if "
            "n.startswith(('organelleverse.operations', 'organelleverse.capabilities', "
            "'organelleverse.assembly', 'organelleverse.annotation'))]))",
        )
        self.assertEqual(json.loads(process.stdout), [])


if __name__ == "__main__":
    unittest.main()
