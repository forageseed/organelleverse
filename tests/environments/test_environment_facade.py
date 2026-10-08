"""RED-phase tests for the public environments facade and CLI (Task 2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from organelleverse import environments as facade
from organelleverse.assembly.environment_contracts import (
    ProviderComponentIdentity,
    ResolvedProvider,
)
from organelleverse.assembly.environment_registry import InstallationRegistry
from organelleverse.tools import environment_cli

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _component(role: str, *, seed: str, version: str | None = "1.0") -> ProviderComponentIdentity:
    return ProviderComponentIdentity(
        role=role,
        kind="executable",
        path=Path(f"/opt/{role}/bin/{role}"),
        sha256=_sha(seed),
        version=version,
    )


def _provider(
    backend_id: str,
    *,
    prefix: Path | None,
    seed: str,
    discovery_source: str = "managed",
) -> ResolvedProvider:
    return ResolvedProvider(
        requested_source="managed",
        discovery_source=discovery_source,  # type: ignore[arg-type]
        carrier="conda",
        platform="linux-64",
        prefix=prefix,
        capability_contract_digest="sha256:" + _sha("contract"),
        components=(_component(backend_id, seed=seed),),
    )


@pytest.fixture
def tool_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the facade's default tool root at an isolated tmp directory."""
    root = tmp_path / "tools"
    monkeypatch.setenv("ORGANELLEVERSE_TOOL_ROOT", str(root))
    return root


def _populate(tool_root: Path, backend_id: str, *, seed: str) -> None:
    registry = InstallationRegistry(tool_root=tool_root)
    registry.register_verified(
        _provider(backend_id, prefix=Path(f"/opt/conda/envs/{backend_id}"), seed=seed)
    )


# ---------------------------------------------------------------------------
# facade: laziness / read-only
# ---------------------------------------------------------------------------


class TestFacadeLazyAndReadOnly:
    def test_list_is_empty_when_registry_absent(self, tool_root: Path) -> None:
        assert facade.list() == ()
        # Lazy: no directory or file materialized just by reading.
        assert not tool_root.exists()

    def test_locate_is_empty_and_lazy_when_registry_absent(self, tool_root: Path) -> None:
        assert facade.locate("oatk") == ()
        assert not tool_root.exists()

    def test_reads_do_not_mutate_tool_root(self, tool_root: Path) -> None:
        _populate(tool_root, "oatk", seed="o")
        snapshot = {p.name for p in tool_root.iterdir()}
        facade.list()
        facade.locate("oatk")
        assert {p.name for p in tool_root.iterdir()} == snapshot


# ---------------------------------------------------------------------------
# facade: record contents
# ---------------------------------------------------------------------------


class TestFacadeRecords:
    def test_list_returns_one_record_per_registered_provider(self, tool_root: Path) -> None:
        _populate(tool_root, "oatk", seed="o")
        _populate(tool_root, "himt", seed="h")

        records = facade.list()
        assert {r.backend_id for r in records} == {"oatk", "himt"}

    def test_record_exposes_full_identity_and_activation(self, tool_root: Path) -> None:
        _populate(tool_root, "oatk", seed="o")
        record = facade.locate("oatk")[0]

        assert record.backend_id == "oatk"
        assert record.prefix == Path("/opt/conda/envs/oatk")
        assert record.executables == (Path("/opt/oatk/bin/oatk"),)
        assert record.version == "1.0"
        assert record.provider_digest.startswith("sha256:")
        assert record.source == "managed"
        assert record.activation == "conda activate /opt/conda/envs/oatk"

    def test_locate_filters_by_backend_and_empty_for_unknown(self, tool_root: Path) -> None:
        _populate(tool_root, "oatk", seed="o")
        assert {r.backend_id for r in facade.locate("oatk")} == {"oatk"}
        assert facade.locate("pmat") == ()

    def test_record_without_prefix_has_empty_activation(self, tool_root: Path) -> None:
        registry = InstallationRegistry(tool_root=tool_root)
        registry.register_verified(_provider("oatk", prefix=None, seed="o"))
        record = facade.locate("oatk")[0]
        assert record.prefix is None
        assert record.activation == ""


# ---------------------------------------------------------------------------
# facade wiring
# ---------------------------------------------------------------------------


class TestFacadeWiring:
    def test_ov_environments_exposes_facade(self, tool_root: Path) -> None:
        import organelleverse as ov

        assert ov.environments is facade
        assert callable(ov.environments.list)
        assert callable(ov.environments.locate)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestEnvironmentCli:
    def test_list_json_emits_record_array(
        self, tool_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _populate(tool_root, "oatk", seed="o")
        rc = environment_cli.main(["list", "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, list)
        assert {entry["backend_id"] for entry in payload} == {"oatk"}
        oatk = next(entry for entry in payload if entry["backend_id"] == "oatk")
        assert oatk["prefix"] == "/opt/conda/envs/oatk"
        assert oatk["activation"] == "conda activate /opt/conda/envs/oatk"

    def test_locate_json_filters_by_backend(
        self, tool_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _populate(tool_root, "oatk", seed="o")
        _populate(tool_root, "himt", seed="h")
        rc = environment_cli.main(["locate", "oatk", "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert {entry["backend_id"] for entry in payload} == {"oatk"}

    def test_list_human_prints_activation_instruction(
        self, tool_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _populate(tool_root, "oatk", seed="o")
        environment_cli.main(["list"])
        out = capsys.readouterr().out
        assert "oatk" in out
        assert "conda activate /opt/conda/envs/oatk" in out

    def test_list_human_empty_registry_prints_nothing(
        self, tool_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        environment_cli.main(["list"])
        assert capsys.readouterr().out == ""
