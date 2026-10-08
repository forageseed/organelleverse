"""Explicit execution-identity authorization for non-core capabilities."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from pydantic import ConfigDict, Field, model_validator

from organelleverse.core.errors import OrganellePermissionError
from organelleverse.operations.spec import StrictSpecModel

from .code_identity import ExecutionIdentity

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_LOCK = threading.RLock()


class TrustedExecution(StrictSpecModel):
    execution_identity: str = Field(pattern=_HASH_RE.pattern)
    capability_id: str = Field(min_length=1)
    bundle_content_hash: str = Field(pattern=_HASH_RE.pattern)
    code_tree_hash: str = Field(pattern=_HASH_RE.pattern)


class TrustDocument(StrictSpecModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
        populate_by_name=True,
    )

    schema_version: Literal["organelleverse.trust.v2"] = Field(
        default="organelleverse.trust.v2", alias="schema"
    )
    trusted: tuple[TrustedExecution, ...] = ()

    @model_validator(mode="after")
    def canonicalize(self) -> TrustDocument:
        ordered = tuple(sorted(self.trusted, key=lambda item: item.execution_identity))
        if ordered != self.trusted:
            object.__setattr__(self, "trusted", ordered)
        return self


def _default_path() -> Path:
    home = Path(os.environ.get("ORGANELLEVERSE_HOME", Path.home() / ".organelleverse"))
    return home / "trust.json"


def _validate_hash(execution_digest: str) -> None:
    if _HASH_RE.fullmatch(execution_digest) is None:
        raise ValueError(f"invalid execution identity digest: {execution_digest!r}")


class TrustStore:
    """Owner-controlled authorization document; cross-process updates are last-writer-wins."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _default_path()

    def _read(self) -> TrustDocument:
        if not self.path.exists():
            return TrustDocument()
        if os.name != "nt" and self.path.stat().st_mode & 0o022:
            raise OrganellePermissionError(
                code="capability.trust_store_unsafe",
                message="trust store is group- or world-writable",
                details={"path": str(self.path), "mode": oct(self.path.stat().st_mode & 0o777)},
            )
        try:
            payload = cast(object, json.loads(self.path.read_text(encoding="utf-8")))
            typed_payload = cast(dict[str, object], payload) if isinstance(payload, dict) else None
            if (
                typed_payload is not None
                and typed_payload.get("schema") == "organelleverse.trust.v1"
            ):
                return TrustDocument()
            return TrustDocument.model_validate(payload)
        except (OSError, ValueError) as error:
            raise OrganellePermissionError(
                code="capability.trust_store_invalid",
                message="trust store is unreadable or invalid",
                details={"path": str(self.path), "reason": str(error)},
            ) from error

    def _write(self, document: TrustDocument) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        payload = (
            json.dumps(
                document.model_dump(mode="json", by_alias=True),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                if os.name != "nt":
                    os.chmod(temporary, 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            if os.name != "nt":
                os.chmod(self.path, 0o600)
        finally:
            if temporary.exists():
                temporary.unlink()

    def is_trusted(self, execution_digest: str) -> bool:
        _validate_hash(execution_digest)
        configured = {
            value.strip()
            for value in os.environ.get("ORGANELLEVERSE_TRUST", "").split(",")
            if _HASH_RE.fullmatch(value.strip()) is not None
        }
        if execution_digest in configured:
            return True
        with _LOCK:
            return any(item.execution_identity == execution_digest for item in self._read().trusted)

    def add(
        self,
        capability_id: str,
        execution_identity: ExecutionIdentity,
    ) -> TrustedExecution:
        if capability_id != execution_identity.capability_id:
            raise ValueError(
                "trusted capability ID does not match the execution identity: "
                f"{capability_id!r} != {execution_identity.capability_id!r}"
            )
        _validate_hash(execution_identity.digest)
        with _LOCK:
            document = self._read()
            by_identity = {item.execution_identity: item for item in document.trusted}
            record = TrustedExecution(
                execution_identity=execution_identity.digest,
                capability_id=capability_id,
                bundle_content_hash=execution_identity.bundle_content_hash,
                code_tree_hash=execution_identity.code_tree_hash,
            )
            by_identity[execution_identity.digest] = record
            self._write(TrustDocument(trusted=tuple(by_identity.values())))
            return record


def trust(
    capability_id: str,
    execution_identity: ExecutionIdentity,
    *,
    store: TrustStore | None = None,
) -> TrustedExecution:
    """Explicitly authorize one exact non-core bundle execution identity."""
    return (store or TrustStore()).add(capability_id, execution_identity)


__all__ = ["TrustDocument", "TrustStore", "TrustedExecution", "trust"]
