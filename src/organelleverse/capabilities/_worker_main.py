# pyright: basic
"""Pure-stdlib entry point for one bundle-local inspection or invocation."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import math
import pathlib
import struct
import sys
import types
import typing

_PROTOCOL = "organelleverse.bundle-worker.v1"

#: The one OrganelleVerse module staged plugin code may import. Served from
#: bytes read out of this script's own package tree (never through ambient
#: finders, so the worker never resolves a different OrganelleVerse than its
#: parent), under a synthetic empty ``organelleverse`` package that replaces
#: the eager root facade inside the worker.
_PROTOCOL_MODULE = "organelleverse.plugin_protocol"
_PROTOCOL_PACKAGE = "organelleverse"
_SYNTHETIC_PACKAGE_SOURCE = b'"""Worker-served synthetic package: plugin protocol access only."""\n'
_MAX_FRAME_BYTES = 8 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 10_000
_MAX_SOURCE_FILE_BYTES = 2 * 1024 * 1024
_MAX_SOURCE_TOTAL_BYTES = 8 * 1024 * 1024
_MAX_SOURCE_FILES = 256
_MAX_SOURCE_CANDIDATES = 512
_MAX_SOURCE_PATH_BYTES = 1024
_MAX_SOURCE_DEPTH = 32
_SOURCE_READ_CHUNK = 64 * 1024
_MAX_ERROR_REASON_CHARS = 4_096
# Skipped for the same reason as in ``code_identity.py``: an installed bundle
# always carries a derived ``__pycache__``. Both snapshots must skip it
# identically or the parent and the worker would hash different trees.
_CACHE_DIRECTORY = "__pycache__"


class _WorkerFailure(Exception):
    def __init__(self, code: str, message: str, **details: object) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


class _PluginExecutionFailure(_WorkerFailure):
    """An exception raised by the plugin callable itself."""


class _PluginResultShape(typing.Protocol):
    """Runtime-checked through the worker-served protocol module."""

    summary: dict[str, object]
    outputs: dict[str, str]
    log_paths: tuple[str, ...]


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _validate_json_structure(value: object) -> None:
    remaining = _MAX_JSON_NODES
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        remaining -= 1
        if remaining < 0 or depth > _MAX_JSON_DEPTH:
            raise ValueError("JSON structure exceeds the worker depth or node bound")
        if item is None or isinstance(item, (bool, str, int)):
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
            continue
        if isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
            continue
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            stack.extend((child, depth + 1) for child in item.values())
            continue
        raise ValueError(f"{type(item).__name__} is not JSON-compatible")


def _loads_json(content: bytes) -> object:
    try:
        value = json.loads(
            content.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
        _validate_json_structure(value)
        if _canonical_json(value) != content:
            raise ValueError("worker request JSON is not in canonical wire form")
        return value
    except (RecursionError, UnicodeDecodeError, ValueError) as error:
        raise _WorkerFailure(
            "capability.worker_frame_invalid",
            "worker request is not bounded canonical JSON",
            reason=str(error),
        ) from error


def _read_exact(handle: typing.BinaryIO, size: int) -> bytes:
    parts: list[bytes] = []
    remaining = size
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            raise _WorkerFailure(
                "capability.worker_frame_invalid",
                "worker request frame ended before its declared length",
            )
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _read_frame(handle: typing.BinaryIO) -> object:
    header = _read_exact(handle, 8)
    size = struct.unpack(">Q", header)[0]
    if size > _MAX_FRAME_BYTES:
        raise _WorkerFailure(
            "capability.worker_frame_too_large",
            "worker request frame exceeds the configured bound",
            max_bytes=_MAX_FRAME_BYTES,
            observed_bytes=size,
        )
    content = _read_exact(handle, size)
    if handle.read(1):
        raise _WorkerFailure(
            "capability.worker_frame_invalid",
            "worker request contains trailing data or a second frame",
        )
    return _loads_json(content)


def _canonical_json(value: object) -> bytes:
    _validate_json_structure(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_frame(handle: typing.BinaryIO, value: object) -> None:
    content = _canonical_json(value)
    if len(content) > _MAX_FRAME_BYTES:
        raise _WorkerFailure(
            "capability.worker_frame_too_large",
            "worker response frame exceeds the configured bound",
            max_bytes=_MAX_FRAME_BYTES,
            observed_bytes=len(content),
        )
    handle.write(struct.pack(">Q", len(content)))
    handle.write(content)
    handle.flush()


def _canonical_hash(entries: dict[str, bytes]) -> str:
    leaves = [[name, hashlib.sha256(content).hexdigest()] for name, content in entries.items()]
    leaves.sort(key=lambda item: item[0])
    canonical = json.dumps(
        leaves,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _bundle_hash(bundle_root_text: str) -> str:
    bundle_root = pathlib.Path(bundle_root_text)
    try:
        resolved_root = bundle_root.resolve(strict=True)
    except OSError as error:
        raise _WorkerFailure(
            "capability.bundle_read_failed",
            "worker cannot resolve the complete bundle tree",
            path=bundle_root_text,
            reason=str(error),
        ) from error
    if not resolved_root.is_dir():
        raise _WorkerFailure(
            "capability.bundle_read_failed",
            "worker bundle root is not a directory",
            path=str(resolved_root),
        )
    try:
        candidates = sorted(resolved_root.rglob("*"), key=lambda item: item.as_posix())
    except OSError as error:
        raise _WorkerFailure(
            "capability.bundle_read_failed",
            "worker cannot enumerate the complete bundle tree",
            path=str(resolved_root),
            reason=str(error),
        ) from error
    leaves: list[list[str]] = []
    for candidate in candidates:
        if candidate.is_symlink():
            raise _WorkerFailure(
                "capability.bundle_path_unsafe",
                "worker bundle tree contains a symbolic link",
                path=str(candidate),
            )
        if not candidate.is_file():
            continue
        try:
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(resolved_root).as_posix()
            digest = hashlib.sha256()
            with resolved.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except (OSError, ValueError) as error:
            raise _WorkerFailure(
                "capability.bundle_read_failed",
                "worker cannot hash a complete bundle file",
                path=str(candidate),
                reason=str(error),
            ) from error
        leaves.append([relative, digest.hexdigest()])
    leaves.sort(key=lambda item: item[0])
    canonical = json.dumps(
        leaves,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _execution_digest(identity: dict[str, object]) -> str:
    payload = {
        name: identity[name]
        for name in (
            "kind",
            "capability_id",
            "bundle_content_hash",
            "code_tree_hash",
            "callable_locator",
            "interpreter",
            "worker_protocol",
        )
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _read_source(path: pathlib.Path) -> bytes:
    content = bytearray()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(_SOURCE_READ_CHUNK)
                if not chunk:
                    break
                content.extend(chunk)
                if len(content) > _MAX_SOURCE_FILE_BYTES:
                    raise _WorkerFailure(
                        "capability.code_source_too_large",
                        "bundle source exceeds the worker per-file bound",
                        path=str(path),
                        max_bytes=_MAX_SOURCE_FILE_BYTES,
                    )
    except _WorkerFailure:
        raise
    except (MemoryError, OSError) as error:
        raise _WorkerFailure(
            "capability.code_read_failed",
            "bundle source could not be read into the bounded worker snapshot",
            path=str(path),
            reason=str(error),
        ) from error
    try:
        return bytes(content)
    except MemoryError as error:
        raise _WorkerFailure(
            "capability.code_snapshot_too_large",
            "bundle source could not be frozen within the worker snapshot bound",
            path=str(path),
            max_bytes=_MAX_SOURCE_TOTAL_BYTES,
        ) from error


def _snapshot_sources(bundle_root_text: str) -> tuple[pathlib.Path, dict[str, bytes]]:
    bundle_root = pathlib.Path(bundle_root_text)
    if not bundle_root.is_absolute():
        raise _WorkerFailure(
            "capability.code_path_unsafe",
            "worker bundle root must be absolute",
            path=bundle_root_text,
        )
    try:
        resolved_bundle = bundle_root.resolve(strict=True)
        code_root = (resolved_bundle / "code").resolve(strict=True)
    except OSError as error:
        raise _WorkerFailure(
            "capability.code_tree_missing",
            "worker cannot resolve the bundle code tree",
            path=bundle_root_text,
            reason=str(error),
        ) from error
    if (resolved_bundle / "code").is_symlink() or not code_root.is_dir():
        raise _WorkerFailure(
            "capability.code_path_unsafe",
            "worker code root must be a real directory",
            path=str(code_root),
        )
    try:
        code_root.relative_to(resolved_bundle)
    except ValueError as error:
        raise _WorkerFailure(
            "capability.code_path_unsafe",
            "worker code root escapes the bundle",
            path=str(code_root),
        ) from error

    candidates: list[pathlib.Path] = []
    try:
        for candidate in code_root.rglob("*"):
            if _CACHE_DIRECTORY in candidate.relative_to(code_root).parts:
                continue
            candidates.append(candidate)
            if len(candidates) > _MAX_SOURCE_CANDIDATES:
                raise _WorkerFailure(
                    "capability.code_snapshot_too_large",
                    "worker code tree contains too many candidate paths",
                    max_candidates=_MAX_SOURCE_CANDIDATES,
                )
        candidates.sort(key=lambda item: item.as_posix())
    except _WorkerFailure:
        raise
    except (MemoryError, OSError) as error:
        raise _WorkerFailure(
            "capability.code_read_failed",
            "worker code tree cannot be enumerated safely",
            reason=str(error),
        ) from error

    sources: dict[str, bytes] = {}
    directories: list[pathlib.Path] = []
    total_bytes = 0
    for candidate in candidates:
        if candidate.is_symlink():
            raise _WorkerFailure(
                "capability.code_path_unsafe",
                "worker code tree contains a symlink",
                path=str(candidate),
            )
        try:
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(code_root)
            stat_result = resolved.stat()
        except (OSError, ValueError) as error:
            raise _WorkerFailure(
                "capability.code_path_unsafe",
                "worker code path cannot be resolved below its root",
                path=str(candidate),
                reason=str(error),
            ) from error
        relative_text = relative.as_posix()
        if (
            len(relative.parts) > _MAX_SOURCE_DEPTH
            or len(relative_text.encode("utf-8")) > _MAX_SOURCE_PATH_BYTES
        ):
            raise _WorkerFailure(
                "capability.code_path_unsafe",
                "worker code path exceeds the supported depth or length",
                path=str(candidate),
            )
        if candidate.is_dir():
            directories.append(resolved)
            continue
        if not candidate.is_file() or not relative_text.endswith(".py"):
            raise _WorkerFailure(
                "capability.code_layout_invalid",
                "worker code tree accepts regular Python source files only",
                path=str(candidate),
            )
        if len(sources) >= _MAX_SOURCE_FILES:
            raise _WorkerFailure(
                "capability.code_snapshot_too_large",
                "worker code tree contains too many source files",
                max_files=_MAX_SOURCE_FILES,
            )
        if stat_result.st_size > _MAX_SOURCE_FILE_BYTES:
            raise _WorkerFailure(
                "capability.code_source_too_large",
                "worker source exceeds the per-file bound",
                path=str(candidate),
                max_bytes=_MAX_SOURCE_FILE_BYTES,
            )
        if total_bytes + stat_result.st_size > _MAX_SOURCE_TOTAL_BYTES:
            raise _WorkerFailure(
                "capability.code_snapshot_too_large",
                "worker sources exceed the total snapshot bound",
                max_bytes=_MAX_SOURCE_TOTAL_BYTES,
            )
        content = _read_source(resolved)
        total_bytes += len(content)
        if total_bytes > _MAX_SOURCE_TOTAL_BYTES:
            raise _WorkerFailure(
                "capability.code_snapshot_too_large",
                "worker sources exceed the total snapshot bound",
                max_bytes=_MAX_SOURCE_TOTAL_BYTES,
            )
        sources[relative_text] = content

    top_level = sorted(code_root.iterdir(), key=lambda item: item.name)
    if (
        len(top_level) != 1
        or top_level[0].is_symlink()
        or not top_level[0].is_dir()
        or not top_level[0].name.isidentifier()
    ):
        raise _WorkerFailure(
            "capability.code_layout_invalid",
            "worker code tree must contain exactly one root package",
        )
    root_name = top_level[0].name
    if root_name in sys.stdlib_module_names or root_name in sys.builtin_module_names:
        raise _WorkerFailure(
            "capability.code_root_collision",
            "worker private root collides with the Python runtime",
            root_package=root_name,
        )
    source_names = set(sources)
    for directory in (top_level[0].resolve(strict=True), *directories):
        relative = directory.relative_to(code_root)
        if any(not part.isidentifier() for part in relative.parts):
            raise _WorkerFailure(
                "capability.code_layout_invalid",
                "worker package directory is not a Python identifier",
                path=str(directory),
            )
        if (relative / "__init__.py").as_posix() not in source_names:
            raise _WorkerFailure(
                "capability.code_namespace_package_unsupported",
                "worker does not support namespace packages",
                path=str(directory),
            )
    module_names: set[str] = set()
    for relative_text in sorted(source_names):
        relative = pathlib.Path(relative_text)
        if relative.name == "__init__.py":
            fullname = ".".join(relative.parent.parts)
        else:
            if not relative.stem.isidentifier():
                raise _WorkerFailure(
                    "capability.code_layout_invalid",
                    "worker module name is not a Python identifier",
                    path=relative_text,
                )
            fullname = ".".join((*relative.parent.parts, relative.stem))
        if fullname in module_names:
            raise _WorkerFailure(
                "capability.code_module_ambiguous",
                "worker sources map multiple files to one module fullname",
                module=fullname,
            )
        module_names.add(fullname)
    return code_root, sources


def _verify_identity(
    expected: dict[str, object],
    sources: dict[str, bytes],
    bundle_content_hash: str,
) -> dict[str, object]:
    actual = {
        "kind": "bundle-local-python-v1",
        "capability_id": expected.get("capability_id"),
        "bundle_content_hash": bundle_content_hash,
        "code_tree_hash": _canonical_hash(sources),
        "callable_locator": expected.get("callable_locator"),
        "interpreter": (
            f"{sys.implementation.name}-{sys.version_info.major}.{sys.version_info.minor}"
        ),
        "worker_protocol": _PROTOCOL,
    }
    actual["digest"] = _execution_digest(actual)
    if actual != expected:
        raise _WorkerFailure(
            "capability.worker_identity_mismatch",
            "worker source snapshot does not match the expected execution identity",
            expected_digest=expected.get("digest"),
            actual_digest=actual["digest"],
            expected_code_hash=expected.get("code_tree_hash"),
            actual_code_hash=actual["code_tree_hash"],
            expected_bundle_hash=expected.get("bundle_content_hash"),
            actual_bundle_hash=bundle_content_hash,
        )
    return actual


class _SourceLoader(importlib.abc.Loader):
    def __init__(self, source: bytes, origin: str, is_package: bool) -> None:
        self._source = source
        self._origin = origin
        self._is_package = is_package

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType | None:
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        module.__file__ = self._origin
        code = compile(self._source, self._origin, "exec", dont_inherit=True)
        exec(code, module.__dict__)


class _ClosedFinder(importlib.abc.MetaPathFinder):
    def __init__(
        self,
        code_root: pathlib.Path,
        sources: dict[str, bytes],
        root_name: str,
    ) -> None:
        self._root = root_name
        self._modules: dict[str, tuple[bytes, str, bool]] = {}
        protocol_path = pathlib.Path(__file__).resolve().parents[1] / "plugin_protocol.py"
        self._modules[_PROTOCOL_PACKAGE] = (
            _SYNTHETIC_PACKAGE_SOURCE,
            str(protocol_path.parent / "__init__.py"),
            True,
        )
        self._modules[_PROTOCOL_MODULE] = (
            protocol_path.read_bytes(),
            str(protocol_path),
            False,
        )
        for relative_text, source in sources.items():
            relative = pathlib.Path(relative_text)
            is_package = relative.name == "__init__.py"
            fullname = (
                ".".join(relative.parent.parts)
                if is_package
                else ".".join((*relative.parent.parts, relative.stem))
            )
            self._modules[fullname] = (
                source,
                str(code_root / relative),
                is_package,
            )

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        top_name = fullname.split(".", 1)[0]
        if top_name == _PROTOCOL_PACKAGE:
            item = self._modules.get(fullname)
            if item is None:
                raise ModuleNotFoundError(
                    f"import {fullname!r} is outside the verified private/stdlib closure",
                    name=fullname,
                )
            source, origin, is_package = item
            loader = _SourceLoader(source, origin, is_package)
            return importlib.util.spec_from_loader(
                fullname,
                loader,
                origin=origin,
                is_package=is_package,
            )
        if top_name == self._root:
            item = self._modules.get(fullname)
            if item is None:
                raise ModuleNotFoundError(
                    f"private module {fullname!r} is absent from the verified source snapshot",
                    name=fullname,
                )
            source, origin, is_package = item
            loader = _SourceLoader(source, origin, is_package)
            return importlib.util.spec_from_loader(
                fullname,
                loader,
                origin=origin,
                is_package=is_package,
            )
        if top_name not in sys.stdlib_module_names and top_name not in sys.builtin_module_names:
            raise ModuleNotFoundError(
                f"import {fullname!r} is outside the verified private/stdlib closure",
                name=fullname,
            )
        return None


def _annotation_token(annotation: object) -> str:
    if annotation is inspect.Parameter.empty:
        return "unsupported"
    exact: dict[object, str] = {
        str: "str",
        pathlib.Path: "path",
        int: "int",
        float: "float",
        bool: "bool",
        list: "list",
        dict: "dict",
        typing.Any: "json",
    }
    if annotation in exact:
        return exact[annotation]
    if isinstance(annotation, str):
        normalized = " ".join(annotation.strip().split())
        strings = {
            "str": "str",
            "Path": "path",
            "pathlib.Path": "path",
            "int": "int",
            "float": "float",
            "bool": "bool",
            "list": "list",
            "dict": "dict",
            "Any": "json",
            "typing.Any": "json",
            "str | None": "str_or_none",
            "None | str": "str_or_none",
            "Path | None": "path_or_none",
            "None | Path": "path_or_none",
            "pathlib.Path | None": "path_or_none",
            "None | pathlib.Path": "path_or_none",
        }
        return strings.get(normalized, "unsupported")
    args = typing.get_args(annotation)
    if len(args) == 2 and type(None) in args:
        other = args[0] if args[1] is type(None) else args[1]
        if other is str:
            return "str_or_none"
        if other is pathlib.Path:
            return "path_or_none"
    return "unsupported"


def _parameter_kind(kind: object) -> str:
    kinds: dict[object, str] = {
        inspect.Parameter.POSITIONAL_ONLY: "positional_only",
        inspect.Parameter.POSITIONAL_OR_KEYWORD: "positional_or_keyword",
        inspect.Parameter.KEYWORD_ONLY: "keyword_only",
    }
    if kind not in kinds:
        raise _WorkerFailure(
            "capability.execution_provider_required",
            "variadic bundle-local callables require an execution provider",
            parameter_kind=str(kind),
        )
    return kinds[kind]


def _json_default(default: object, annotation: str) -> object:
    if isinstance(default, pathlib.Path) and annotation in {"path", "path_or_none"}:
        return str(default)
    _validate_json_structure(default)
    return default


def _inspect_parameters(
    function: typing.Callable[..., object],
) -> list[dict[str, object]]:
    try:
        signature = inspect.signature(function, eval_str=False)
    except (TypeError, ValueError) as error:
        raise _WorkerFailure(
            "capability.execution_provider_required",
            "bundle-local callable signature cannot be inspected safely",
            reason=str(error),
        ) from error
    parameters: list[dict[str, object]] = []
    for parameter in signature.parameters.values():
        annotation = _annotation_token(parameter.annotation)
        required = parameter.default is inspect.Parameter.empty
        parameters.append(
            {
                "name": parameter.name,
                "kind": _parameter_kind(parameter.kind),
                "required": required,
                "default": (None if required else _json_default(parameter.default, annotation)),
                "annotation": annotation,
            }
        )
    return parameters


#: The fixed plugin_protocol callable shape: ``run(inputs, outputs,
#: parameters, context)``. Inspection pins these exact tokens so a plugin
#: whose signature drifts fails verification, never runtime.
_PLUGIN_SIGNATURE_TOKENS: tuple[tuple[str, str], ...] = (
    ("inputs", "dict"),
    ("outputs", "dict"),
    ("parameters", "dict"),
    ("context", "plugin_context"),
)


def _enforce_plugin_signature(
    function: typing.Callable[..., object], inspected: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Validate the standard plugin callable shape and emit its fixed tokens."""

    expected_names = [name for name, _ in _PLUGIN_SIGNATURE_TOKENS]
    if len(inspected) != len(expected_names) or [
        item["name"] for item in inspected
    ] != expected_names:
        raise _WorkerFailure(
            "capability.plugin_signature_invalid",
            "plugin callable must be run(inputs, outputs, parameters, context)",
        )
    for item, (name, token) in zip(inspected, _PLUGIN_SIGNATURE_TOKENS, strict=True):
        if name != "context" and item["annotation"] != token:
            raise _WorkerFailure(
                "capability.plugin_signature_invalid",
                f"plugin parameter {name!r} must be annotated as a dict",
                parameter=name,
            )
    context_annotation = list(inspect.signature(function).parameters.values())[3].annotation
    annotation_name = (
        context_annotation
        if isinstance(context_annotation, str)
        else getattr(context_annotation, "__name__", "")
    )
    if not annotation_name.endswith("PluginContext"):
        raise _WorkerFailure(
            "capability.plugin_signature_invalid",
            "plugin context parameter must be annotated PluginContext",
            parameter="context",
        )
    return [
        {
            "name": name,
            "kind": item["kind"],
            "required": True,
            "default": None,
            "annotation": token,
        }
        for item, (name, token) in zip(inspected, _PLUGIN_SIGNATURE_TOKENS, strict=True)
    ]


def _root_name(sources: dict[str, bytes]) -> str:
    return next(iter(sources)).split("/", 1)[0]


def _resolve_private_callable(
    locator: object,
    *,
    code_root: pathlib.Path,
    sources: dict[str, bytes],
    root_name: str,
) -> typing.Callable[..., object]:
    """Resolve one ``module:attribute`` locator inside the closed bundle root.

    Both the run locator and the declared score locator resolve through this
    one helper and therefore share the same ``_ClosedFinder`` import policy
    and private-root rule.
    """
    if not isinstance(locator, str) or ":" not in locator:
        raise _WorkerFailure(
            "capability.code_locator_invalid",
            "worker callable locator is invalid",
        )
    module_name, attribute_name = locator.split(":", 1)
    if module_name.split(".", 1)[0] != root_name:
        raise _WorkerFailure(
            "capability.code_locator_outside_root",
            "worker callable locator is outside the private root",
        )
    finder = _ClosedFinder(code_root, sources, root_name)
    sys.meta_path.insert(0, finder)
    module = importlib.import_module(module_name)
    function = getattr(module, attribute_name, None)
    if not callable(function):
        raise _WorkerFailure(
            "capability.binding_invalid",
            "worker callable locator does not resolve to a callable",
            callable_locator=locator,
        )
    return typing.cast(typing.Callable[..., object], function)


def _enforce_score_signature(scorer: typing.Callable[..., object]) -> None:
    """Pin the v2 score hook shape: ``score(result: PluginResult) -> float``."""
    try:
        signature = inspect.signature(scorer, eval_str=False)
    except (TypeError, ValueError) as error:
        raise _WorkerFailure(
            "capability.plugin_score_invalid",
            "declared plugin score callable signature cannot be inspected",
            reason=str(error),
        ) from error
    parameters = list(signature.parameters.values())
    if (
        len(parameters) != 1
        or parameters[0].kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        or parameters[0].default is not inspect.Parameter.empty
    ):
        raise _WorkerFailure(
            "capability.plugin_score_invalid",
            "declared plugin score callable must take exactly one required "
            "positional-or-keyword PluginResult parameter",
        )
    annotation = parameters[0].annotation
    annotation_name = (
        annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    )
    if not annotation_name.endswith("PluginResult"):
        raise _WorkerFailure(
            "capability.plugin_score_invalid",
            "declared plugin score parameter must be annotated PluginResult",
        )
    return_annotation = signature.return_annotation
    if return_annotation is not float and return_annotation != "float":
        raise _WorkerFailure(
            "capability.plugin_score_invalid",
            "declared plugin score callable must explicitly return float",
        )


def _resolve_score_callable(
    optimization: object,
    *,
    code_root: pathlib.Path,
    sources: dict[str, bytes],
    root_name: str,
) -> typing.Callable[..., object]:
    """Resolve the declared score callable through the bundle-local boundary."""
    locator = optimization.get("score_locator") if isinstance(optimization, dict) else None
    try:
        scorer = _resolve_private_callable(
            locator, code_root=code_root, sources=sources, root_name=root_name
        )
    except _WorkerFailure as error:
        raise _WorkerFailure(
            "capability.plugin_score_invalid",
            "declared plugin score locator does not resolve to a bundle-local callable",
            score_locator=locator if isinstance(locator, str) else None,
        ) from error
    _enforce_score_signature(scorer)
    return scorer


def _validate_declared_scorer(
    request: dict[str, object], code_root: pathlib.Path, sources: dict[str, bytes]
) -> None:
    """Pin the declared score hook during worker inspection.

    Verification runs inspection before trust, so a missing, non-callable, or
    wrongly shaped score callable fails here with
    ``capability.plugin_score_invalid`` and never reaches runtime.
    """
    contract = typing.cast(dict[str, object], request["contract"])
    binding = contract.get("binding")
    if not (isinstance(binding, dict) and binding.get("argument_mode") == "plugin_protocol"):
        return
    optimization = contract.get("optimization")
    if optimization is None:
        return
    _resolve_score_callable(
        optimization, code_root=code_root, sources=sources, root_name=_root_name(sources)
    )


def _load_callable(
    request: dict[str, object], code_root: pathlib.Path, sources: dict[str, bytes]
) -> tuple[typing.Callable[..., object], list[dict[str, object]]]:
    identity = typing.cast(dict[str, object], request["execution_identity"])
    function = _resolve_private_callable(
        identity.get("callable_locator"),
        code_root=code_root,
        sources=sources,
        root_name=_root_name(sources),
    )
    inspected = _inspect_parameters(function)
    contract = typing.cast(dict[str, object], request["contract"])
    binding = contract.get("binding")
    if isinstance(binding, dict) and binding.get("argument_mode") == "plugin_protocol":
        inspected = _enforce_plugin_signature(function, inspected)
    return function, inspected


def _decode_value(value: object, annotation: str) -> object:
    if annotation in {"path", "path_or_none"} and value is not None:
        if not isinstance(value, str):
            raise _WorkerFailure(
                "capability.binding_invalid",
                "path parameter did not receive a JSON string",
            )
        return pathlib.Path(value)
    return value


def _invoke_plugin(
    function: typing.Callable[..., object],
    request: dict[str, object],
    *,
    code_root: pathlib.Path,
    sources: dict[str, bytes],
) -> dict[str, object]:
    """Invoke the fixed v2 plugin protocol inside one managed staging tree."""

    contract = typing.cast(dict[str, object], request["contract"])
    binding = typing.cast(dict[str, object], contract["binding"])
    declared = binding.get("parameters")
    declared_outputs = contract.get("outputs")
    supplied = request.get("parameters")
    if not isinstance(declared, list) or not isinstance(declared_outputs, list):
        raise _WorkerFailure("capability.binding_invalid", "plugin contract ports are invalid")
    if not isinstance(supplied, dict):
        raise _WorkerFailure("capability.binding_invalid", "worker parameters must be an object")
    if not all(isinstance(item, dict) for item in declared) or not all(
        isinstance(item, dict) for item in declared_outputs
    ):
        raise _WorkerFailure("capability.binding_invalid", "plugin contract ports are invalid")

    output_parameter_names = {
        output.get("parameter") for output in declared_outputs if isinstance(output.get("parameter"), str)
    }
    declared_names = {item.get("name") for item in declared if isinstance(item.get("name"), str)}
    allowed_names = declared_names - output_parameter_names
    extras = set(supplied) - allowed_names
    if extras:
        raise _WorkerFailure(
            "capability.binding_invalid",
            "worker received undeclared plugin parameters",
            parameters=sorted(extras),
        )

    inputs: dict[str, object] = {}
    parameters: dict[str, object] = {}
    for item in declared:
        name = item.get("name")
        codec = item.get("codec")
        source = item.get("source", "agent")
        if not isinstance(name, str) or not isinstance(codec, str) or source != "agent":
            raise _WorkerFailure("capability.binding_invalid", "plugin binding parameter is invalid")
        if name in output_parameter_names:
            continue
        if name in supplied:
            value = supplied[name]
        else:
            schema = item.get("json_schema")
            if isinstance(schema, dict) and "default" in schema:
                value = schema["default"]
            else:
                raise _WorkerFailure(
                    "capability.binding_invalid",
                    f"plugin parameter {name!r} was not supplied",
                    parameter=name,
                )
        if codec == "json":
            parameters[name] = value
        elif codec in {"path", "directory"}:
            if not isinstance(value, str):
                raise _WorkerFailure(
                    "capability.binding_invalid",
                    f"plugin {codec} parameter must be a JSON string",
                    parameter=name,
                )
            inputs[name] = value
        else:
            raise _WorkerFailure(
                "capability.execution_provider_required",
                f"plugin parameter codec {codec!r} requires an execution provider",
                parameter=name,
            )

    staging_root = pathlib.Path(typing.cast(str, request["staging_root"]))
    outputs_root = staging_root / "outputs"
    allocated: dict[str, pathlib.Path] = {}
    for output in declared_outputs:
        name = output.get("name")
        kind = output.get("kind")
        if not isinstance(name, str) or kind not in {"file", "directory"}:
            raise _WorkerFailure("capability.binding_invalid", "plugin output declaration is invalid")
        destination = outputs_root / name
        if kind == "directory":
            destination.mkdir(parents=True, exist_ok=False)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
        allocated[name] = destination

    protocol = importlib.import_module(_PROTOCOL_MODULE)
    context = protocol.PluginContext(
        capability_id=typing.cast(str, request["operation_id"]),
        run_id=typing.cast(str, request["run_id"]),
        work_dir=staging_root,
        log_path=staging_root / "plugin.log",
    )
    try:
        result = function(
            inputs=inputs,
            outputs={name: str(path) for name, path in allocated.items()},
            parameters=parameters,
            context=context,
        )
    except Exception as error:
        raise _PluginExecutionFailure(
            "capability.plugin_execution_failed",
            "plugin callable raised an exception",
            exception_type=type(error).__name__,
            reason=_bounded_exception_reason(error),
        ) from error
    result_type = getattr(protocol, "PluginResult", None)
    if not isinstance(result_type, type) or not isinstance(result, result_type):
        raise _WorkerFailure(
            "capability.plugin_result_invalid",
            "plugin callable must return PluginResult",
        )
    plugin_result = typing.cast(_PluginResultShape, result)
    produced = plugin_result.outputs
    if set(produced) != set(allocated):
        raise _WorkerFailure(
            "capability.plugin_result_invalid",
            "plugin result outputs must name exactly the declared outputs",
        )
    relative_outputs: dict[str, str] = {}
    for output in declared_outputs:
        name = typing.cast(str, output["name"])
        kind = typing.cast(str, output["kind"])
        produced_path = pathlib.Path(produced[name])
        expected_path = allocated[name]
        if kind == "file":
            valid = produced_path == expected_path and produced_path.is_file()
        else:
            try:
                produced_path.relative_to(expected_path)
                valid = produced_path.is_dir() and any(
                    path.is_file() for path in produced_path.rglob("*")
                )
            except ValueError:
                valid = False
        if not valid:
            raise _WorkerFailure(
                "capability.plugin_result_invalid",
                f"plugin output {name!r} was not produced in its allocated destination",
                output=name,
            )
        relative_outputs[name] = produced_path.relative_to(staging_root).as_posix()

    relative_logs: list[str] = []
    for raw_log_path in plugin_result.log_paths:
        if not isinstance(raw_log_path, str):
            raise _WorkerFailure(
                "capability.plugin_result_invalid",
                "plugin log paths must be strings",
            )
        log_path = pathlib.Path(raw_log_path)
        try:
            log_path.relative_to(staging_root)
        except ValueError as error:
            raise _WorkerFailure(
                "capability.plugin_result_invalid",
                "plugin log path must be inside the staging tree",
            ) from error
        if not log_path.is_file():
            raise _WorkerFailure(
                "capability.plugin_result_invalid",
                "plugin log path must name an existing file",
            )
        relative_logs.append(log_path.relative_to(staging_root).as_posix())

    score: float | None = None
    metric_key = typing.cast(str, protocol.PLUGIN_OPTIMIZATION_SCORE_METRIC)
    if metric_key in plugin_result.summary:
        raise _WorkerFailure(
            "capability.plugin_score_reserved",
            f"plugin summary key {metric_key!r} is reserved",
        )
    optimization = contract.get("optimization")
    if optimization is not None:
        scorer = _resolve_score_callable(
            optimization,
            code_root=code_root,
            sources=sources,
            root_name=_root_name(sources),
        )
        try:
            raw_score = scorer(result)
        except Exception as error:
            raise _WorkerFailure(
                "capability.plugin_score_failed",
                "declared plugin score callable raised an exception",
                exception_type=type(error).__name__,
                reason=_bounded_exception_reason(error),
            ) from error
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise _WorkerFailure(
                "capability.plugin_score_invalid",
                "declared plugin score callable must return a finite number",
            )
        try:
            score = float(raw_score)
        except OverflowError as error:
            raise _WorkerFailure(
                "capability.plugin_score_invalid",
                "declared plugin score callable must return a finite number",
            ) from error
        if not math.isfinite(score):
            raise _WorkerFailure(
                "capability.plugin_score_invalid",
                "declared plugin score callable must return a finite number",
            )

    return {
        "summary": plugin_result.summary,
        "outputs": relative_outputs,
        "log_paths": relative_logs,
        "score": score,
    }


def _invoke_function(
    function: typing.Callable[..., object],
    inspected: list[dict[str, object]],
    request: dict[str, object],
    *,
    code_root: pathlib.Path,
    sources: dict[str, bytes],
) -> object:
    contract = typing.cast(dict[str, object], request["contract"])
    binding = contract.get("binding")
    if isinstance(binding, dict) and binding.get("argument_mode") == "plugin_protocol":
        return _invoke_plugin(function, request, code_root=code_root, sources=sources)
    if not isinstance(binding, dict) or binding.get("argument_mode") != "named_parameters":
        raise _WorkerFailure(
            "capability.execution_provider_required",
            "worker supports named_parameters bindings only",
        )
    declared = binding.get("parameters", [])
    if not isinstance(declared, list):
        raise _WorkerFailure("capability.binding_invalid", "worker binding parameters are invalid")
    supplied = request.get("parameters")
    if not isinstance(supplied, dict):
        raise _WorkerFailure("capability.binding_invalid", "worker parameters must be an object")
    inspected_by_name = {item["name"]: item for item in inspected}
    values: dict[str, object] = {}
    declared_names: set[str] = set()
    for item in declared:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise _WorkerFailure(
                "capability.binding_invalid", "worker parameter binding is invalid"
            )
        name = typing.cast(str, item["name"])
        declared_names.add(name)
        if item.get("source", "agent") != "agent":
            raise _WorkerFailure(
                "capability.execution_provider_required",
                "environment-sourced parameters require an execution provider",
                parameter=name,
            )
        if name in supplied:
            inspected_item = inspected_by_name.get(name)
            if inspected_item is None:
                raise _WorkerFailure(
                    "capability.binding_invalid",
                    "declared parameter is absent from the callable signature",
                    parameter=name,
                )
            values[name] = _decode_value(
                supplied[name], typing.cast(str, inspected_item["annotation"])
            )
    extras = set(supplied) - declared_names
    if extras:
        raise _WorkerFailure(
            "capability.binding_invalid",
            "worker received undeclared parameters",
            parameters=sorted(extras),
        )
    core_name = binding.get("core_input_parameter")
    if core_name is not None:
        if not isinstance(core_name, str):
            raise _WorkerFailure("capability.binding_invalid", "core input binding is invalid")
        values[core_name] = request.get("input")
    elif request.get("input") is not None:
        raise _WorkerFailure(
            "capability.binding_invalid",
            "worker received an undeclared core input",
        )

    positional: list[object] = []
    keywords: dict[str, object] = {}
    for item in inspected:
        name = typing.cast(str, item["name"])
        if name not in values:
            continue
        if item["kind"] == "positional_only":
            positional.append(values[name])
        else:
            keywords[name] = values[name]
    try:
        signature = inspect.signature(function, eval_str=False)
        bound = signature.bind(*positional, **keywords)
        bound.apply_defaults()
    except TypeError as error:
        raise _WorkerFailure(
            "capability.binding_invalid",
            "worker could not bind the declared invocation",
            reason=str(error),
        ) from error
    return function(*bound.args, **bound.kwargs)


def _bounded_exception_reason(error: BaseException) -> str:
    try:
        reason = str(error)
    except BaseException as formatting_error:
        reason = f"<{type(formatting_error).__name__} while formatting exception>"
    return reason[:_MAX_ERROR_REASON_CHARS]


def _require_request(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _WorkerFailure(
            "capability.worker_protocol_invalid",
            "worker request must be a JSON object",
        )
    allowed_fields = {
        "protocol",
        "request_id",
        "operation_id",
        "mode",
        "execution_identity",
        "bundle_root",
        "contract",
        "worker_parameters",
        "input",
        "parameters",
        "run_id",
        "staging_root",
    }
    extras = set(value) - allowed_fields
    if extras:
        raise _WorkerFailure(
            "capability.worker_protocol_invalid",
            "worker request contains fields outside the closed protocol envelope",
            fields=sorted(extras),
        )
    required_strings = (
        "protocol",
        "request_id",
        "operation_id",
        "mode",
        "bundle_root",
        "run_id",
        "staging_root",
    )
    for name in required_strings:
        if not isinstance(value.get(name), str):
            raise _WorkerFailure(
                "capability.worker_protocol_invalid",
                f"worker request field {name!r} must be a string",
            )
    if value["protocol"] != _PROTOCOL or value["mode"] not in {"inspect", "invoke"}:
        raise _WorkerFailure(
            "capability.worker_protocol_invalid",
            "worker request protocol or mode is unsupported",
        )
    identity = value.get("execution_identity")
    contract = value.get("contract")
    frozen_parameters = value.get("worker_parameters")
    if (
        not isinstance(identity, dict)
        or not isinstance(contract, dict)
        or not isinstance(frozen_parameters, list)
    ):
        raise _WorkerFailure(
            "capability.worker_protocol_invalid",
            "worker request identity, contract, or signature is invalid",
        )
    if identity.get("capability_id") != value["operation_id"]:
        raise _WorkerFailure(
            "capability.worker_operation_mismatch",
            "worker request operation does not match its execution identity",
        )
    staging = pathlib.Path(typing.cast(str, value["staging_root"]))
    if not staging.is_absolute() or staging.is_symlink() or not staging.is_dir():
        raise _WorkerFailure(
            "capability.worker_protocol_invalid",
            "worker staging root must be an existing private directory",
        )
    return value


def _run(request: dict[str, object]) -> dict[str, object]:
    identity = typing.cast(dict[str, object], request["execution_identity"])
    bundle_root = typing.cast(str, request["bundle_root"])
    bundle_content_hash = _bundle_hash(bundle_root)
    code_root, sources = _snapshot_sources(bundle_root)
    actual_identity = _verify_identity(identity, sources, bundle_content_hash)
    function, parameters = _load_callable(request, code_root, sources)
    if request["mode"] == "inspect":
        _validate_declared_scorer(request, code_root, sources)
        return {
            "protocol": _PROTOCOL,
            "request_id": request["request_id"],
            "operation_id": request["operation_id"],
            "mode": "inspect",
            "execution_identity": actual_identity,
            "status": "ok",
            "parameters": parameters,
            "value": None,
            "artifact_paths": [],
            "error": None,
        }
    if request["worker_parameters"] != parameters:
        raise _WorkerFailure(
            "capability.schema_drift",
            "worker callable signature differs from the frozen verification tokens",
        )
    contract = typing.cast(dict[str, object], request["contract"])
    binding = contract.get("binding")
    is_plugin = isinstance(binding, dict) and binding.get("argument_mode") == "plugin_protocol"
    try:
        result = _invoke_function(function, parameters, request, code_root=code_root, sources=sources)
    except _PluginExecutionFailure:
        raise
    except _WorkerFailure:
        raise
    except Exception as error:
        if not is_plugin:
            raise
        raise _WorkerFailure(
            "capability.plugin_result_invalid",
            "plugin result violates the declared protocol",
            exception_type=type(error).__name__,
            reason=_bounded_exception_reason(error),
        ) from error
    try:
        _validate_json_structure(result)
    except (RecursionError, TypeError, ValueError) as error:
        raise _WorkerFailure(
            "capability.plugin_result_invalid"
            if is_plugin
            else "capability.worker_protocol_invalid",
            "plugin result is not bounded JSON" if is_plugin else "worker result is not bounded JSON",
            reason=_bounded_exception_reason(error),
        ) from error
    return {
        "protocol": _PROTOCOL,
        "request_id": request["request_id"],
        "operation_id": request["operation_id"],
        "mode": "invoke",
        "execution_identity": actual_identity,
        "status": "ok",
        "parameters": [],
        "value": result,
        "artifact_paths": [],
        "error": None,
    }


def _error_response(request: dict[str, object], error: _WorkerFailure) -> dict[str, object]:
    return {
        "protocol": _PROTOCOL,
        "request_id": request["request_id"],
        "operation_id": request["operation_id"],
        "mode": request["mode"],
        "execution_identity": request["execution_identity"],
        "status": "error",
        "parameters": [],
        "value": None,
        "artifact_paths": [],
        "error": {
            "code": error.code,
            "message": error.message,
            "details": error.details,
        },
    }


def main() -> int:
    protocol_stdout = sys.stdout.buffer
    sys.stdout = sys.stderr
    try:
        request = _require_request(_read_frame(sys.stdin.buffer))
    except _WorkerFailure as error:
        sys.stderr.write(f"{error.code}: {error.message}\n")
        return 2
    try:
        response = _run(request)
    except _WorkerFailure as error:
        response = _error_response(request, error)
    except BaseException as error:
        response = _error_response(
            request,
            _WorkerFailure(
                "capability.worker_execution_failed",
                "bundle-local worker execution failed",
                exception_type=type(error).__name__,
                reason=_bounded_exception_reason(error),
            ),
        )
    try:
        _write_frame(protocol_stdout, response)
    except _WorkerFailure as error:
        sys.stderr.write(f"{error.code}: {error.message}\n")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
