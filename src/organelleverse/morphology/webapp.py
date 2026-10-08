"""Gradio web UI for OrgSegNet plant-EM organelle segmentation.

A local web app (mirroring the Plantorganelle Hunter web version at
cropopen.com/#/Cell) where you upload a TEM image and get back the
segmentation mask, overlay, per-class counts, and per-object morphometrics.

The segmentation backend calls OrgSegNet via the real mmseg API
(:func:`.orgseg.default_executor`). OrgSegNet must be installed in an
``orgseg`` conda env (py3.8 + torch1.13 + mmcv2.0rc4 + mmseg1.0 + the repo
pip-installed); when it is absent the UI still opens but the "Segment" button
returns the install guide instead of crashing.

Start with::

    import organelleverse as ov

    ov.morph.serve(orgseg_root="/path/to/OrgSegNet", device="cuda:0")
    # -> opens http://127.0.0.1:7860
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .install import check_backend, install_hint
from .measure import ORGSEG_CLASSES, ORGSEG_PALETTE, measure, overlay, summarize
from .orgseg import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, default_executor

__all__ = ["serve", "build_app"]


def _segment_one(
    image: np.ndarray | None,
    image_path: str | None,
    *,
    config: str,
    checkpoint: str,
    device: str,
    threshold_chloro: float,
    threshold_mito: float,
    threshold_vac: float,
    threshold_nuc: float,
    pixel_size_um: float,
    watershed_mito: bool,
    out_dir: Path,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[list[Any]], str, str]:
    """Run OrgSegNet on one image and return (orig, mask_rgb, overlay, rows, counts_json, status).

    When OrgSegNet is not installed, returns the install hint as the status and
    empty outputs.
    """
    if image is None and not image_path:
        return None, None, None, [], "", "No image uploaded."

    loc = check_backend("orgseg", scan_envs=True)
    if not loc.get("installed"):
        return None, None, None, [], "", install_hint("orgseg")

    # Write the uploaded image to a temp path if given as an array.
    from PIL import Image

    img_path = image_path
    if img_path is None:
        img_path = str(out_dir / "upload.png")
        Image.fromarray(image).save(img_path)

    try:
        label_paths = default_executor(
            config,
            checkpoint,
            device,
            [img_path],
            str(out_dir),
        )
    except Exception as exc:  # noqa: BLE001
        return None, None, None, [], "", f"OrgSegNet inference failed: {exc}"

    if not label_paths:
        return None, None, None, [], "", "OrgSegNet produced no output."

    label_path = label_paths[0]
    lab = np.array(Image.open(str(label_path)))
    if lab.ndim == 3:
        lab = lab[..., 0]

    # Load source image for overlay + intensity.
    src = np.array(Image.open(img_path))
    if src.ndim == 2:
        src_disp = np.stack([src] * 3, axis=-1)
    else:
        src_disp = src[..., :3] if src.shape[-1] >= 3 else src

    # Colorized mask.
    mask_rgb = np.zeros(lab.shape + (3,), dtype=np.uint8)
    for cls_idx, rgb in ORGSEG_PALETTE.items():
        mask_rgb[lab == cls_idx] = rgb

    # Overlay.
    ovl = overlay(src, lab, alpha=0.5)

    # Morphometrics.
    m = measure(
        lab,
        intensity_image=src,
        pixel_size_um=pixel_size_um or None,
        watershed_classes=(2,) if watershed_mito else (),
    )
    rows = [
        [
            o["class_name"],
            o["area_px"],
            o["perimeter"],
            o["circularity"],
            o["eccentricity"],
            o["equivalent_diameter"],
            o["mean_intensity"],
        ]
        for o in m["per_object"]
    ]
    counts = dict(m["counts"])
    counts_str = ", ".join(f"{k}: {v}" for k, v in counts.items() if v) or "none"
    return (
        src_disp,
        mask_rgb,
        ovl,
        rows,
        counts_str,
        f"Segmented: {counts_str} ({m['n_total']} objects)",
    )


def build_app(
    *,
    config: str = DEFAULT_CONFIG,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda:0",
) -> Any:
    """Build (but do not launch) the Gradio Blocks web app.

    Returns the ``gr.Blocks`` instance. Use :func:`serve` to build + launch.
    """
    import gradio as gr
    import tempfile

    out_dir = Path(tempfile.mkdtemp(prefix="morph_web_"))

    with gr.Blocks(title="OrganelleVerse · Plant EM Segmentation") as demo:
        gr.Markdown(
            "# 🌿 OrganelleVerse — Plant EM Organelle Segmentation\n"
            "Upload a TEM image; OrgSegNet (Plantorganelle Hunter, "
            "Nat. Plants 2023) segments Chloroplast / Mitochondria / Vacuole / "
            "Nucleus and reports per-object morphometrics.\n\n"
            f"Config: `{config}`  |  Checkpoint: `{checkpoint}`  |  Device: `{device}`"
        )
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(label="Upload TEM image", type="filepath")
                with gr.Accordion("Thresholds & options", open=False):
                    t_chloro = gr.Slider(0, 1, 0.5, step=0.05, label="Chloroplast threshold")
                    t_mito = gr.Slider(0, 1, 0.5, step=0.05, label="Mitochondria threshold")
                    t_vac = gr.Slider(0, 1, 0.5, step=0.05, label="Vacuole threshold")
                    t_nuc = gr.Slider(0, 1, 0.5, step=0.05, label="Nucleus threshold")
                    px = gr.Number(value=0.0, label="Pixel size (µm/px, 0=off)")
                    ws = gr.Checkbox(value=True, label="Watershed-split mitochondria")
                btn = gr.Button("🧪 Segment", variant="primary")
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Overlay"):
                        ovl_out = gr.Image(label="Segmentation overlay")
                    with gr.Tab("Mask"):
                        mask_out = gr.Image(label="Class-colorized mask")
                    with gr.Tab("Original"):
                        orig_out = gr.Image(label="Source image")
                counts_out = gr.Textbox(label="Per-class counts")
                tbl = gr.Dataframe(
                    headers=[
                        "class",
                        "area_px",
                        "perimeter",
                        "circularity",
                        "eccentricity",
                        "eq_diameter",
                        "mean_intensity",
                    ],
                    label="Per-object morphometrics",
                    wrap=True,
                )
                status = gr.Markdown()

        btn.click(
            _segment_one,
            inputs=[
                img_in,
                img_in,
                gr.State(config),
                gr.State(checkpoint),
                gr.State(device),
                t_chloro,
                t_mito,
                t_vac,
                t_nuc,
                px,
                ws,
                gr.State(str(out_dir)),
            ],
            outputs=[orig_out, mask_out, ovl_out, tbl, counts_out, status],
        )
    return demo


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    config: str | Path | None = None,
    checkpoint: str | Path | None = None,
    device: str = "cuda:0",
    orgseg_root: str | Path | None = None,
) -> Any:
    """Launch the Gradio web app for OrgSegNet segmentation.

    Parameters
    ----------
    host, port
        Bind address (default http://127.0.0.1:7860).
    share
        If True, create a public Gradio share link.
    config
        MMSeg config path (default OrgSeg_PlantCell_768x512.py).
    checkpoint
        ``.pth`` checkpoint (default OrgSegNet_iter_Version1.pth).
    device
        Compute device (``"cuda:0"`` or ``"cpu"``).
    orgseg_root
        Path to the OrgSegNet repo, used to resolve default config/checkpoint.

    Returns
    -------
    gr.Blocks
        The launched Blocks app (also blocks until the server stops).
    """
    cfg = str(config) if config else str(Path(orgseg_root or ".") / DEFAULT_CONFIG)
    ckpt = str(checkpoint) if checkpoint else str(Path(orgseg_root or ".") / DEFAULT_CHECKPOINT)
    app = build_app(config=cfg, checkpoint=ckpt, device=device)
    app.launch(server_name=host, server_port=port, share=share, prevent_thread_lock=False)
    return app
