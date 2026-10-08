from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from organelleverse.capabilities.code_identity import ExecutionIdentity, inspect_bundle_code
from organelleverse.capabilities.hashing import hash_bundle
from organelleverse.capabilities.trust import TrustStore, trust
from organelleverse.core.errors import OrganellePermissionError


def _identity(
    *,
    bundle_hash: str = "sha256:" + "1" * 64,
    code_hash: str = "sha256:" + "2" * 64,
) -> ExecutionIdentity:
    payload = {
        "kind": "bundle-local-python-v1",
        "capability_id": "thirdparty.demo",
        "bundle_content_hash": bundle_hash,
        "code_tree_hash": code_hash,
        "callable_locator": "private_pkg.impl:run",
        "interpreter": "cpython-3.13",
        "worker_protocol": "organelleverse.bundle-worker.v1",
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return ExecutionIdentity(
        capability_id="thirdparty.demo",
        bundle_content_hash=bundle_hash,
        code_tree_hash=code_hash,
        callable_locator="private_pkg.impl:run",
        interpreter="cpython-3.13",
        digest=f"sha256:{hashlib.sha256(canonical).hexdigest()}",
    )


def _write_bundle_code(bundle_root: Path) -> Path:
    package = bundle_root / "code" / "private_pkg"
    package.mkdir(parents=True)
    (bundle_root / "capability.toml").write_text("contract='one'\n", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "impl.py").write_text(
        "from .helper import value\ndef run():\n    return value()\n", encoding="utf-8"
    )
    helper = package / "helper.py"
    helper.write_text("def value():\n    return 1\n", encoding="utf-8")
    return helper


def test_trust_is_keyed_by_composite_execution_digest(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "trust.json")
    trusted = _identity()
    changed_contract = _identity(
        bundle_hash="sha256:" + "4" * 64,
    )

    trust("thirdparty.demo", trusted, store=store)

    assert store.is_trusted(trusted.digest) is True
    assert store.is_trusted(changed_contract.digest) is False


def test_changed_helper_invalidates_execution_trust(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"
    helper = _write_bundle_code(bundle_root)
    before = inspect_bundle_code(
        bundle_root,
        capability_id="thirdparty.demo",
        bundle_content_hash=hash_bundle(bundle_root),
        callable_locator="private_pkg.impl:run",
    )
    store = TrustStore(tmp_path / "state" / "trust.json")
    trust("thirdparty.demo", before, store=store)

    helper.write_text("def value():\n    return 2\n", encoding="utf-8")
    after = inspect_bundle_code(
        bundle_root,
        capability_id="thirdparty.demo",
        bundle_content_hash=hash_bundle(bundle_root),
        callable_locator="private_pkg.impl:run",
    )

    assert store.is_trusted(after.digest) is False


def test_trust_file_is_canonical_owner_only_json(tmp_path: Path) -> None:
    path = tmp_path / "trust.json"
    store = TrustStore(path)
    identity = _identity(code_hash="sha256:" + "a" * 64)
    trust("thirdparty.demo", identity, store=store)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "organelleverse.trust.v2"
    assert payload["trusted"] == [
        {
            "bundle_content_hash": identity.bundle_content_hash,
            "capability_id": identity.capability_id,
            "code_tree_hash": identity.code_tree_hash,
            "execution_identity": identity.digest,
        }
    ]
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    assert path.read_bytes().endswith(b"\n")


def test_group_or_world_writable_trust_store_is_refused(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX permission bits are unavailable")
    path = tmp_path / "trust.json"
    path.write_text('{"schema":"organelleverse.trust.v2","trusted":[]}', encoding="utf-8")
    path.chmod(0o666)

    with pytest.raises(OrganellePermissionError) as captured:
        TrustStore(path).is_trusted("sha256:" + "a" * 64)

    assert captured.value.code == "capability.trust_store_unsafe"


def test_v1_trust_file_and_old_bundle_hash_environment_do_not_authorize_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    path = tmp_path / "trust.json"
    path.write_text(
        json.dumps(
            {
                "schema": "organelleverse.trust.v1",
                "trusted": [
                    {
                        "bundle_content_hash": identity.bundle_content_hash,
                        "capability_ids": [identity.capability_id],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    if os.name != "nt":
        path.chmod(0o600)
    monkeypatch.setenv("ORGANELLEVERSE_TRUST", identity.bundle_content_hash)

    assert TrustStore(path).is_trusted(identity.digest) is False

    monkeypatch.setenv("ORGANELLEVERSE_TRUST", identity.digest)
    assert TrustStore(path).is_trusted(identity.digest) is True


def test_capability_id_mismatch_is_rejected_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "trust.json"
    with pytest.raises(ValueError):
        trust(
            "other.capability",
            _identity(),
            store=TrustStore(path),
        )
    assert not path.exists()


@pytest.mark.parametrize(
    "digest",
    ("", "sha256:not-a-digest", "sha256:" + "A" * 64, "md5:" + "0" * 32),
)
def test_is_trusted_rejects_a_malformed_execution_digest(tmp_path: Path, digest: str) -> None:
    store = TrustStore(tmp_path / "trust.json")

    with pytest.raises(ValueError, match="invalid execution identity digest"):
        store.is_trusted(digest)
