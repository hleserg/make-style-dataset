"""Stage 3 — inpaint the masked bubble/SFX regions out of each panel.

Reads panels from ``01_panels`` and their paired removal masks from ``02_masks``
(written by Stage 2), and writes bubble-free panels to ``03_inpainted``. The
stage iterates over **masks**: each mask pairs to a panel by basename
(``02_masks/foo.png`` ↔ ``01_panels/foo``). Panels that Stage 2 routed to
``manual_review`` (over-coverage / unreadable) therefore have no mask and are
naturally left out of the auto flow; a human handles them. Panels with an empty
(all-black) mask — clean panels that needed no masking — pass straight through
unchanged.

The default backend is **LaMa** run via ONNX (``Carve/LaMa-ONNX``, Apache-2.0),
which needs no diffusion/torch stack. The ONNX session is the only heavy/optional
piece: it is imported lazily inside :class:`LamaInpainter` (``onnxruntime`` lives
in the ``gpu`` dependency-group) and hidden behind the :class:`Inpainter`
protocol, so the pure tensor pre/post-processing and the orchestration are unit
tested without it — the ``pure-core-lazy-backend`` pattern shared with
``panels.py``/``bubbles.py``. The inpainted result is composited back **only**
inside the masked region, so original line-art outside the bubbles is preserved
exactly (a style dataset only needs the text gone, not a repainted page).

Output names are deterministic (``<panel-stem>.png``) so re-runs overwrite
rather than duplicate.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from make_style_dataset.observability import tag_component
from make_style_dataset.stages.base import Stage, StageContext, StageResult

if TYPE_CHECKING:
    import numpy as np

    from make_style_dataset.config import Settings

NAME = "inpaint"
SUMMARY = "Inpaint masked speech bubbles out of each panel."
COMPONENT = "stage:inpaint"

#: ONNX Big-LaMa weights (Apache-2.0). Repo file + pinned commit for a
#: reproducible, supply-chain-safe download (mirrors the Stage 2 model pin).
LAMA_MODEL_REPO = "Carve/LaMa-ONNX"
LAMA_MODEL_FILE = "lama_fp32.onnx"
LAMA_MODEL_REVISION = "c3c0c9e468934d62e79c329e35d82dd09ff8c444"

#: Grayscale level above which a mask pixel counts as "inpaint here".
MASK_THRESHOLD = 127

#: The pinned ONNX model takes a fixed 512x512 input; used as a fallback if the
#: model's declared input shape can't be read.
LAMA_INPUT_SIZE = (512, 512)


class Inpainter(Protocol):
    """Fills masked regions of a panel with plausible background (LaMa's role)."""

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return ``image`` (BGR) with the ``mask`` (255 = remove) regions repainted."""
        ...


# PLAYBOOK-START
# id: pure-core-lazy-backend
# title: Pure policy core behind a lazily-imported heavy backend
# status: draft
# category: testability
# tags: [testing, dependency-injection, coverage]
# Split a stage into pure functions over plain arrays (tensor pre/post,
# compositing) and a thin adapter around a heavy/optional backend (here an
# ONNX Runtime session) that is imported lazily and hidden behind a Protocol.
# Only the model load + session.run stay uncovered; all the array math is unit
# tested. Shared with panels.py and bubbles.py.
# PLAYBOOK-END
class LamaInpainter:
    """Big-LaMa via ONNX Runtime; the session is built lazily on first use."""

    def __init__(self) -> None:
        self._session: Any = None
        self._providers: list[str] = []

    def _get_session(self) -> Any:  # pragma: no cover - needs onnxruntime + model download
        """Download the pinned ONNX weights and open a Runtime session (once)."""
        if self._session is None:
            import onnxruntime  # pyright: ignore[reportMissingImports]
            from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]

            weights = hf_hub_download(
                LAMA_MODEL_REPO, filename=LAMA_MODEL_FILE, revision=LAMA_MODEL_REVISION
            )
            # Prefer CUDA; onnxruntime silently falls back to CPU if it cannot
            # init, so callers should check ``.get_providers()`` to know which ran.
            self._session = onnxruntime.InferenceSession(
                weights, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self._providers = self._session.get_providers()
        return self._session

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Run LaMa and composite the repainted pixels into the masked region."""
        session = self._get_session()
        size = _model_input_size(session) or LAMA_INPUT_SIZE
        feed = lama_inputs(image, mask, size)
        raw = session.run(None, feed)[0]
        painted = lama_output_to_bgr(raw, width=image.shape[1], height=image.shape[0])
        return composite(image, painted, mask)


def make_inpainter(settings: Settings) -> Inpainter:
    """Return the inpainter backend named by ``settings.inpaint_backend``.

    Only ``"lama"`` is implemented; any other value raises so a typo or a
    not-yet-wired backend (e.g. a future SD/ComfyUI option) fails loudly.
    """
    backend = settings.inpaint_backend.strip().lower()
    if backend == "lama":
        return LamaInpainter()
    raise NotImplementedError(
        f"inpaint backend {backend!r} is not implemented; "
        f"set APP_INPAINT_BACKEND=lama (the only supported backend)."
    )


# --- Pure tensor pre/post-processing (no onnxruntime needed) ---------------


def lama_inputs(
    image: np.ndarray, mask: np.ndarray, size: tuple[int, int]
) -> dict[str, np.ndarray]:
    """Build LaMa's ``{"image", "mask"}`` feed from a BGR panel and its mask.

    Both inputs are resized to the model's ``(height, width)`` ``size``; the
    image becomes RGB, scaled to ``[0, 1]`` and laid out as NCHW float32. The
    mask is resized with nearest-neighbour and re-binarised so its edges stay
    hard (no gray bleed into the inpaint region).
    """
    import cv2
    import numpy as np

    target_h, target_w = size
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    binary = (mask > MASK_THRESHOLD).astype(np.uint8)
    binary = cv2.resize(binary, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    image_chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]
    mask_chw = binary.astype(np.float32)[np.newaxis, np.newaxis, ...]
    return {"image": image_chw, "mask": mask_chw}


def lama_output_to_bgr(raw: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Turn LaMa's NCHW float output into a ``height`` x ``width`` BGR uint8 image.

    LaMa emits values around ``[0, 255]`` that can stray slightly out of range,
    so we clip before casting (a bare cast would wrap and speckle the result).
    """
    import cv2
    import numpy as np

    array = np.asarray(raw)
    while array.ndim > 3:  # drop any leading batch dims
        array = array[0]
    hwc = np.transpose(array, (1, 2, 0))  # CHW -> HWC (RGB)
    rgb = np.clip(hwc, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if bgr.shape[0] != height or bgr.shape[1] != width:
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    return bgr


def composite(original: np.ndarray, painted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Take ``painted`` pixels only where ``mask`` is set, ``original`` elsewhere.

    Keeps the panel's line-art byte-identical outside the masked region, so only
    the bubble/SFX areas change.
    """
    import numpy as np

    selected = (mask > MASK_THRESHOLD)[:, :, np.newaxis]
    return np.where(selected, painted, original)


# --- I/O helpers -----------------------------------------------------------


def _decode_bgr(path: Path) -> np.ndarray | None:
    """Decode an image to a BGR array via a byte buffer (OS-agnostic, non-ASCII safe)."""
    import cv2
    import numpy as np

    buffer = np.fromfile(path, dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _decode_mask(path: Path) -> np.ndarray | None:
    """Decode a mask to a single-channel grayscale array via a byte buffer."""
    import cv2
    import numpy as np

    buffer = np.fromfile(path, dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)


def _write_png(array: np.ndarray, path: Path) -> None:
    """Save a uint8 BGR (or grayscale) array as PNG via PIL (never ``cv2.imwrite``)."""
    from PIL import Image

    mode = "L" if array.ndim == 2 else "RGB"
    pixels = array if array.ndim == 2 else array[:, :, ::-1]  # BGR -> RGB for color
    Image.fromarray(pixels, mode=mode).save(path)


def _model_input_size(session: object) -> tuple[int, int] | None:  # pragma: no cover
    """Return the model's fixed (H, W) input dims, or ``None`` when dynamic."""
    shape = session.get_inputs()[0].shape  # type: ignore[attr-defined]
    height, width = shape[2], shape[3]
    if isinstance(height, int) and isinstance(width, int):
        return height, width
    return None


def mask_is_empty(mask: np.ndarray) -> bool:
    """Return ``True`` when no pixel is set (a clean panel that needs no inpainting)."""
    import numpy as np

    return int(np.count_nonzero(mask)) == 0


# --- Orchestration ---------------------------------------------------------


def iter_masks(masks_dir: Path) -> list[Path]:
    """Return mask PNGs under ``masks_dir`` in stable (sorted) order."""
    if not masks_dir.is_dir():
        return []
    return sorted(
        path for path in masks_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    )


def _find_panel(panels_dir: Path, stem: str) -> Path | None:
    """Find the panel whose stem matches a mask, tolerant of the panel suffix."""
    matches = sorted(path for path in panels_dir.iterdir() if path.is_file() and path.stem == stem)
    return matches[0] if matches else None


def inpaint_panel(
    mask_path: Path,
    *,
    inpainter: Inpainter,
    panels_dir: Path,
    out_dir: Path,
) -> bool:
    """Inpaint one panel from its mask, or pass/skip it; return whether one was written.

    Pairs ``mask_path`` with its panel; skips (``False``) when the panel is
    missing (Stage 2 routed it) or unreadable. An empty mask copies the panel
    through unchanged; a non-empty mask is inpainted. Output is ``<stem>.png``.
    """
    panel_path = _find_panel(panels_dir, mask_path.stem)
    if panel_path is None:
        return False
    panel = _decode_bgr(panel_path)
    mask = _decode_mask(mask_path)
    if panel is None or mask is None:
        return False

    out_path = out_dir / f"{mask_path.stem}.png"
    if mask_is_empty(mask):
        _write_png(panel, out_path)
    else:
        _write_png(inpainter.inpaint(panel, mask), out_path)
    return True


def run(ctx: StageContext) -> StageResult:
    """Inpaint every masked panel from ``02_masks`` into ``03_inpainted``."""
    tag_component(COMPONENT)
    out = ctx.workspace.inpainted
    out.mkdir(parents=True, exist_ok=True)

    inpainter = make_inpainter(ctx.settings)
    produced = 0
    for mask_path in iter_masks(ctx.workspace.masks):
        if inpaint_panel(
            mask_path, inpainter=inpainter, panels_dir=ctx.workspace.panels, out_dir=out
        ):
            produced += 1
    return StageResult(name=NAME, output_dir=out, produced=produced)


STAGE = Stage(
    name=NAME,
    summary=SUMMARY,
    component=COMPONENT,
    flag="run_inpaint",
    output=lambda ws, _s: ws.inpainted,
    run=run,
)
