"""Pinned microscopy models and an explicit, offline-capable weight bundle.

Importing this catalog imports neither torch nor a model runtime. Weight
downloads are explicit; segmentation never downloads or substitutes a model.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
import json
import os
import hashlib

_CP_REVISION = "7c61431b5fbb078f3296754bd15d9f51b320f837"
_CP_WEIGHTS = {
    "cpsam_v2": "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667",
    "cpdino": "fad0dda262554922da1c5ab0e0004b00c1b14b8a48bea6781b89222567577573",
    "cpdino-vitb": "3ed4c06a3963ab13ff377d4e2957174aaaf637434eb031ae9931fcbfaf9a217f",
}
_EM_WEIGHTS = {
    "vit_b_em_organelles": ("noisy-ox/1.2", "vit_b", "f3bf2ed83d691456bae2c3f9a05fb438", "bb6398956a6b0132c26b631c14f95ce2"),
    "vit_l_em_organelles": ("humorous-crab/1.2", "vit_l", "810b084b6e51acdbf760a993d8619f2d", "334877640bfdaaabce533e3252a17294"),
}


def list_models() -> list[dict]:
    """Return model IDs, pinned runtimes, modality, label semantics and sources.

    These are models supported by OV, not claims of installed weights or
    accuracy on a user's images. Cellpose models label generic objects;
    micro-SAM EM models label organelle instances without organelle classes.
    """
    rows = []
    for name, digest in _CP_WEIGHTS.items():
        rows.append(dict(
            id=name, backend="cellpose", package="cellpose", runtime_version="4.2.1.1",
            modality="light microscopy; generic cells/objects, validate organelle-specific use",
            mask_kind="instance", dimensions=[2, 3], license="BSD-3-Clause",
            source="https://huggingface.co/mouseland/cellpose-sam", revision=_CP_REVISION,
            license_url="https://github.com/MouseLand/cellpose/blob/main/LICENSE",
            runtime_dependencies=["dinov3"] if name.startswith("cpdino") else [],
            backbone_revision="6876159a11b4df116f30f667f8c9888617df0751" if name.startswith("cpdino") else None,
            backbone_license="https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md" if name.startswith("cpdino") else None,
            files=[dict(name=name, url=f"https://huggingface.co/mouseland/cellpose-sam/resolve/{_CP_REVISION}/{name}", digest="sha256:"+digest)],
        ))
    for name, (record, encoder, digest, decoder_digest) in _EM_WEIGHTS.items():
        base = f"https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/{record}/files"
        rows.append(dict(
            id=name, backend="micro_sam", package="micro-sam", runtime_version="1.8.14",
            modality="electron microscopy; organelle instances, no semantic classes",
            mask_kind="instance", dimensions=[2, 3], license="CC-BY-4.0 (weights)",
            source=f"https://bioimage.io/#/artifacts/{record}", revision=record,
            license_url="https://creativecommons.org/licenses/by/4.0/",
            files=[dict(name=encoder+".pt", url=base+f"/{encoder}.pt", digest="xxh128:"+digest),
                   dict(name=encoder+"_decoder.pt", url=base+f"/{encoder}_decoder.pt", digest="xxh128:"+decoder_digest)],
        ))
    rows.append(dict(
        id="orgsegnet-v1", backend="orgseg", package="mmsegmentation", runtime_version="1.0.0",
        modality="plant transmission electron microscopy; Chloroplast/Mitochondria/Vacuole/Nucleus",
        mask_kind="semantic", dimensions=[2], license="CC-BY-4.0 (weights)",
        source="https://zenodo.org/records/8419877", revision="8419877",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        files=[dict(name="OrgSegNet_iter_Version1.pth", url="https://zenodo.org/api/records/8419877/files/OrgSegNet_iter_Version1.pth/content", digest="md5:9c91a0f5e1eb90beac7e320c4e1ae3e7")],
    ))
    return rows


def model_guide() -> str:
    """Read the microscopy guide shipped with an ordinary installed OV wheel."""
    from importlib.resources import files
    return files("organelleverse").joinpath("morphology/MODELS.md").read_text(encoding="utf-8")


def _profile(model_id: str) -> dict:
    for row in list_models():
        if row["id"] == model_id:
            return row
    raise ValueError(f"Unknown microscopy model {model_id!r}; choose an ID from list_models()")


def _root(directory: str | Path | None = None) -> Path:
    if directory is not None:
        return Path(directory)
    return Path(os.environ.get("ORGANELLEVERSE_MODEL_DIR", Path.home()/".cache/organelleverse/models"))


def model_status(model_id: str, directory: str | Path | None = None) -> dict:
    """Report weights and runtime separately without importing torch or downloading."""
    row = _profile(model_id)
    folder = _root(directory)/model_id
    paths = [folder/item["name"] for item in row["files"]]
    try:
        installed = version(row["package"])
    except PackageNotFoundError:
        installed = None
    missing = [name for name in row.get("runtime_dependencies", []) if find_spec(name) is None]
    return dict(model=row, directory=str(folder), files=[str(p) for p in paths], missing_packages=missing,
                weights_present=all(p.is_file() for p in paths), runtime_version=installed,
                runtime_matches=installed == row["runtime_version"] and not missing)


def download_models(directory: str | Path, model_ids: list[str] | None = None) -> Path:
    """Build a portable weight bundle, checking publisher-provided digests.

    Requires ``pooch`` and ``xxhash`` (the ``morphology-models`` extra).
    ``HF_ENDPOINT`` explicitly selects a Hugging Face mirror if needed.
    No runtime is installed. Download on the compute server, not through a
    workstation relay. Sources and attribution travel with the bundle.
    """
    import pooch
    rows = [_profile(name) for name in model_ids] if model_ids is not None else list_models()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        folder = root/row["id"]
        folder.mkdir(exist_ok=True)
        for item in row["files"]:
            url = item["url"]
            endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
            if url.startswith("https://huggingface.co/"):
                url = endpoint + url[len("https://huggingface.co"):]
            pooch.retrieve(url, known_hash=item["digest"], fname=item["name"], path=folder,
                           downloader=pooch.HTTPDownloader(timeout=120), progressbar=True)
        (folder/"MODEL.json").write_text(json.dumps(row, indent=2)+"\n", encoding="utf-8")
    manifest = root/"models.json"
    existing = json.loads(manifest.read_text()) if manifest.is_file() else []
    combined = {row["id"]: row for row in existing}
    combined.update({row["id"]: row for row in rows})
    manifest.write_text(json.dumps(list(combined.values()), indent=2)+"\n", encoding="utf-8")
    (root/"README.txt").write_text(
        "OrganelleVerse microscopy weights. models.json records upstream sources, versions and licenses.\n"
        "Weights retain their upstream licenses; OV's license does not replace them.\n"
        "OrgSegNet: Feng et al., Nature Plants 2023, doi:10.1038/s41477-023-01527-5.\n"
        "micro-SAM: Archit et al., Nature Methods 2025, doi:10.1038/s41592-024-02580-4.\n"
        "Cellpose: https://github.com/MouseLand/cellpose (BSD-3-Clause).\n"
        "Install each runtime separately using its pinned version. Do not mix legacy OrgSegNet with modern runtimes.\n",
        encoding="utf-8")
    return root


def make_executor(model_id: str, directory: str | Path | None = None, *, is_3d_stack: bool = False, voxel_size_um=None):
    """Resolve an offline native executor; preserve the existing four-argument hook."""
    state = model_status(model_id, directory)
    row = state["model"]
    if row["backend"] == "orgseg":
        raise ValueError("OrgSegNet requires its upstream source/configuration; use backend='orgseg' and the existing executor interface")
    if not state["weights_present"]:
        raise FileNotFoundError(f"Missing {model_id} weights in {state['directory']}; use download_models() explicitly")
    if not state["runtime_matches"]:
        raise RuntimeError(f"{model_id} requires {row['package']}=={row['runtime_version']}; found {state['runtime_version']!r}, missing {state['missing_packages']}")
    paths = [Path(p) for p in state["files"]]
    for path, weight in zip(paths, row["files"]):
        algorithm, expected = weight["digest"].split(":", 1)
        if algorithm == "xxh128":
            import xxhash
            factory = xxhash.xxh128
        else:
            factory = algorithm
        with path.open("rb") as source:
            observed = hashlib.file_digest(source, factory).hexdigest()
        if observed != expected:
            raise ValueError(f"{model_id} weight differs from the pinned publisher checkpoint: {path}")
    def execute(_model_type, device, image_paths, out_dir):
        if row["backend"] == "cellpose":
            from .cellpose import execute_cellpose
            return execute_cellpose(paths[0], device, image_paths, out_dir, is_3d_stack=is_3d_stack, voxel_size_um=voxel_size_um)
        from .microsam import execute_offline_micro_sam
        return execute_offline_micro_sam(model_id, paths[0], paths[1], device, image_paths, out_dir, is_3d_stack=is_3d_stack)
    return execute


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "download"])
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--model", action="append")
    args = parser.parse_args()
    if args.command == "list":
        print(json.dumps(list_models(), indent=2))
    elif args.directory is None:
        parser.error("download requires --directory")
    else:
        print(download_models(args.directory, args.model))
