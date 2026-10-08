"""RED-phase tests for the persistent installation registry (Task 2)."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from organelleverse.assembly.environment_contracts import (
    ProviderComponentIdentity,
    ResolvedProvider,
)
from organelleverse.assembly.environment_registry import (
    InstallationRegistry,
    RegisteredComponent,
    RegisteredProvider,
    default_tool_root,
)
from organelleverse.assembly.environment_resolver import (
    _default_registry_lookup,  # pyright: ignore[reportPrivateUsage]
)

# ---------------------------------------------------------------------------
# helpers
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
    *,
    backend_id: str = "oatk",
    prefix: Path | None = Path("/opt/conda/envs/oatk"),
    seed: str = "oatk",
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


# ---------------------------------------------------------------------------
# default_tool_root
# ---------------------------------------------------------------------------


class TestDefaultToolRoot:
    def test_respects_organelleverse_tool_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ORGANELLEVERSE_TOOL_ROOT", "/custom/tools")
        assert default_tool_root() == Path("/custom/tools")

    def test_uses_xdg_data_home_when_tool_root_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ORGANELLEVERSE_TOOL_ROOT", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", "/custom/xdg")
        assert default_tool_root() == Path("/custom/xdg/organelleverse/tools")

    def test_falls_back_to_local_share(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("ORGANELLEVERSE_TOOL_ROOT", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert default_tool_root() == tmp_path / ".local" / "share" / "organelleverse" / "tools"

    def test_tool_root_takes_precedence_over_xdg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ORGANELLEVERSE_TOOL_ROOT", "/env/tools")
        monkeypatch.setenv("XDG_DATA_HOME", "/xdg/tools")
        assert default_tool_root() == Path("/env/tools")


# ---------------------------------------------------------------------------
# constructor / read-only guarantees
# ---------------------------------------------------------------------------


class TestConstructorDoesNotWrite:
    def test_constructor_creates_nothing(self, tmp_path: Path) -> None:
        root = tmp_path / "tools"
        InstallationRegistry(tool_root=root)
        assert not root.exists()

    def test_list_on_empty_registry_returns_empty_and_writes_nothing(self, tmp_path: Path) -> None:
        root = tmp_path / "tools"
        registry = InstallationRegistry(tool_root=root)
        assert registry.list() == ()
        assert registry.locate("oatk") == ()
        # No files created, no directory created.
        assert not root.exists()

    def test_locate_is_read_only_when_registry_absent(self, tmp_path: Path) -> None:
        root = tmp_path / "tools"
        registry = InstallationRegistry(tool_root=root)
        registry.locate("oatk")
        assert not root.exists()


# ---------------------------------------------------------------------------
# register_verified
# ---------------------------------------------------------------------------


class TestRegisterVerified:
    def test_first_register_creates_tool_root_lock_and_registry(self, tmp_path: Path) -> None:
        root = tmp_path / "tools"
        registry = InstallationRegistry(tool_root=root)

        record = registry.register_verified(_provider())

        assert isinstance(record, RegisteredProvider)
        assert root.is_dir()
        assert (root / "environments.json").is_file()
        # Lock sibling exists after a write.
        assert (root / "environments.lock").is_file()
        # No leftover temporary sibling files.
        assert not any(p.name.startswith("environments.json.") for p in root.iterdir())

    def test_register_records_provider_digest_and_components(self, tmp_path: Path) -> None:
        registry = InstallationRegistry(tool_root=tmp_path / "tools")
        provider = _provider()
        record = registry.register_verified(provider)

        assert record.backend_id == "oatk"
        assert record.provider_digest == provider.provider_digest
        assert record.capability_contract_digest == provider.capability_contract_digest
        assert record.platform == "linux-64"
        assert record.source == "managed"
        assert record.prefix == Path("/opt/conda/envs/oatk")
        assert record.version == "1.0"
        assert len(record.components) == 1
        comp = record.components[0]
        assert isinstance(comp, RegisteredComponent)
        assert comp.role == "oatk"
        assert comp.sha256 == provider.components[0].sha256

    def test_register_is_idempotent_by_provider_digest(self, tmp_path: Path) -> None:
        registry = InstallationRegistry(tool_root=tmp_path / "tools")
        provider = _provider()

        first = registry.register_verified(provider)
        second = registry.register_verified(provider)

        assert first == second
        records = registry.list()
        assert len(records) == 1

    def test_register_distinct_providers_both_recorded(self, tmp_path: Path) -> None:
        registry = InstallationRegistry(tool_root=tmp_path / "tools")
        a = _provider(backend_id="oatk", seed="a")
        b = _provider(backend_id="himt", seed="b", prefix=Path("/opt/conda/envs/himt"))

        registry.register_verified(a)
        registry.register_verified(b)

        records = registry.list()
        assert {r.backend_id for r in records} == {"oatk", "himt"}

    def test_register_retains_distinct_versions_of_same_backend(self, tmp_path: Path) -> None:
        registry = InstallationRegistry(tool_root=tmp_path / "tools")
        registry.register_verified(_provider(seed="original"))

        # Same backend, different component hash -> different provider_digest.
        registry.register_verified(_provider(seed="updated"))
        records = registry.locate("oatk")
        assert len(records) == 2
        assert {record.components[0].sha256 for record in records} == {
            _sha("original"),
            _sha("updated"),
        }

    def test_provider_identity_is_stable_across_discovery_sources(self) -> None:
        managed = _provider(discovery_source="managed")
        rediscovered = managed.model_copy(update={"discovery_source": "registry"})
        rediscovered = ResolvedProvider.model_validate(
            rediscovered.model_dump(exclude={"provider_digest"})
        )
        assert rediscovered.provider_digest == managed.provider_digest

    def test_non_object_registry_is_rejected_cleanly(self, tmp_path: Path) -> None:
        root = tmp_path / "tools"
        root.mkdir()
        (root / "environments.json").write_text("[]")

        with pytest.raises(ValueError, match="JSON object"):
            InstallationRegistry(tool_root=root).list()


class TestLocateAndList:
    def test_list_returns_all_records(self, tmp_path: Path) -> None:
        registry = InstallationRegistry(tool_root=tmp_path / "tools")
        registry.register_verified(_provider(backend_id="oatk", seed="o"))
        registry.register_verified(_provider(backend_id="himt", seed="h"))

        records = registry.list()
        assert {r.backend_id for r in records} == {"oatk", "himt"}

    def test_locate_filters_by_backend(self, tmp_path: Path) -> None:
        registry = InstallationRegistry(tool_root=tmp_path / "tools")
        registry.register_verified(_provider(backend_id="oatk", seed="o"))
        registry.register_verified(_provider(backend_id="himt", seed="h"))

        oatk = registry.locate("oatk")
        assert len(oatk) == 1
        assert oatk[0].backend_id == "oatk"
        assert registry.locate("pmat") == ()

    def test_default_resolver_lookup_reads_verified_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "tools"
        monkeypatch.setenv("ORGANELLEVERSE_TOOL_ROOT", str(root))
        provider = _provider()
        InstallationRegistry(tool_root=root).register_verified(provider)

        result = _default_registry_lookup("oatk")

        assert result is not None
        registered = result["providers"][0]  # type: ignore[index]
        component = registered["components"]["oatk"]  # type: ignore[index]
        assert component["path"] == "/opt/oatk/bin/oatk"
        assert component["registered_sha256"] == provider.components[0].sha256

    def test_list_is_read_only_after_register(self, tmp_path: Path) -> None:
        root = tmp_path / "tools"
        registry = InstallationRegistry(tool_root=root)
        registry.register_verified(_provider())

        snapshot = {p.name for p in root.iterdir()}
        registry.list()
        registry.locate("oatk")
        assert {p.name for p in root.iterdir()} == snapshot


# ---------------------------------------------------------------------------
# concurrency + atomicity
# ---------------------------------------------------------------------------


class TestConcurrencyAndAtomicity:
    def test_concurrent_registrations_all_recorded(self, tmp_path: Path) -> None:
        registry = InstallationRegistry(tool_root=tmp_path / "tools")
        backends = ["oatk", "himt", "pmat", "getorganelle", "probe"]

        def register_one(name: str) -> None:
            registry.register_verified(
                _provider(backend_id=name, seed=name, prefix=Path(f"/opt/{name}"))
            )

        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(register_one, backends))

        records = {r.backend_id for r in registry.list()}
        assert records == set(backends)
        # Single canonical registry file, no leftover temps.
        assert (tmp_path / "tools" / "environments.json").is_file()
        assert not any(
            p.name.startswith("environments.json.") for p in (tmp_path / "tools").iterdir()
        )

    def test_registry_file_is_canonical_json(self, tmp_path: Path) -> None:
        import json

        registry = InstallationRegistry(tool_root=tmp_path / "tools")
        registry.register_verified(_provider())
        raw = (tmp_path / "tools" / "environments.json").read_text()
        data = json.loads(raw)
        assert data["schema_version"] == "organelleverse.environment-registry.v1"
        assert isinstance(data["providers"], list)
        assert len(data["providers"]) == 1


# ---------------------------------------------------------------------------
# no global side effects
# ---------------------------------------------------------------------------


class TestNoGlobalSideEffects:
    def test_register_writes_only_under_tool_root(self, tmp_path: Path) -> None:
        root = tmp_path / "tools"
        registry = InstallationRegistry(tool_root=root)
        registry.register_verified(_provider())

        written: list[Path] = []
        for path in root.rglob("*"):
            if path.is_file():
                written.append(path)
        # Only registry json + lock live under tool root.
        names = {p.name for p in written}
        assert names <= {"environments.json", "environments.lock"}
        # Nothing escaped the tool root.
        assert all(root in p.parents for p in written)
