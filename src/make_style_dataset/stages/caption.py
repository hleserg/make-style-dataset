"""Stage 5 — auto-caption clean panels and lay out the kohya dataset.

Reads clean panels from ``04_clean`` and writes the final kohya-ready dataset to
``05_dataset/<repeats>_<trigger>/``: each image is copied in as ``<stem>.png``
with a sibling ``<stem>.txt`` caption. Captions follow the **Style Locker**
strategy — the ``trigger_token`` comes first, then WD14 booru **content** tags
(objects, pose, shot, background). Style is deliberately *not* described in
words so it binds to the trigger; whatever is left undescribed is what the LoRA
learns as the style.

The default tagger is **WD14 ViT v3** (SmilingWolf) run via ONNX (Apache-2.0,
torch-free), behind the :class:`Tagger` protocol. ``onnxruntime`` +
``huggingface_hub`` are imported lazily inside :class:`Wd14Tagger`, so the pure
tag-vocabulary parsing, tag selection, caption building and I/O are unit tested
without them — the ``pure-core-lazy-backend`` pattern shared with
``inpaint.py``/``bubbles.py``. Only the model load + ``session.run`` are
uncovered. Output names are deterministic, so re-runs overwrite rather than
duplicate.

Auto-captions always need a human proof-read on a real dataset (a wrong tag
teaches a wrong association); this stage produces the first draft.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from make_style_dataset.media import image_files
from make_style_dataset.observability import tag_component
from make_style_dataset.stages.base import Stage, StageContext, StageResult

if TYPE_CHECKING:
    import numpy as np

    from make_style_dataset.config import Settings
    from make_style_dataset.workspace import Workspace

NAME = "caption"
SUMMARY = "Caption clean panels and lay out the kohya-ready dataset folder."
COMPONENT = "stage:caption"

#: WD14 ViT v3 tagger (SmilingWolf, Apache-2.0). Repo file + tags CSV + pinned
#: commit for a reproducible, supply-chain-safe download (mirrors S2/S3).
WD14_MODEL_REPO = "SmilingWolf/wd-vit-tagger-v3"
WD14_MODEL_FILE = "model.onnx"
WD14_TAGS_FILE = "selected_tags.csv"
WD14_MODEL_REVISION = "7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f"

#: The model's fixed square input edge (config.json: img_size=448).
WD14_IMAGE_SIZE = 448

#: ``selected_tags.csv`` category integer for general/content tags (vs 9=rating,
#: 4=character) — only these go into a style dataset's content caption.
CATEGORY_GENERAL = 0

#: Default probability cutoff for keeping a general tag (kohya's --thresh).
DEFAULT_CAPTION_THRESHOLD = 0.35


class Tagger(Protocol):
    """Extracts booru content tags from a panel (WD14's role)."""

    def tag(self, image: np.ndarray) -> list[str]:
        """Return general/content booru tags for a BGR ``image``."""
        ...


# PLAYBOOK-START
# id: pure-core-lazy-backend
# title: Pure policy core behind a lazily-imported heavy backend
# status: draft
# category: testability
# tags: [testing, dependency-injection, coverage]
# Split a stage into pure functions over plain data (vocabulary parsing, tag
# selection, caption building) and a thin adapter around a heavy/optional
# backend (here a WD14 ONNX Runtime session) imported lazily behind a Protocol.
# Only the model load + session.run stay uncovered; all the tag logic is unit
# tested. Shared with panels.py, bubbles.py and inpaint.py.
# PLAYBOOK-END
class Wd14Tagger:
    """SmilingWolf WD14 ViT v3 via ONNX Runtime; session built lazily on first use."""

    def __init__(self, threshold: float = DEFAULT_CAPTION_THRESHOLD) -> None:
        self._threshold = threshold
        self._session: Any = None
        self._general_names: list[str] = []
        self._general_mask: Any = None

    def _load(self) -> None:  # pragma: no cover - needs onnxruntime + model download
        """Download the pinned ONNX weights + tags CSV and open a session (once)."""
        if self._session is not None:
            return
        import onnxruntime  # pyright: ignore[reportMissingImports]
        from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]

        weights = hf_hub_download(
            WD14_MODEL_REPO, filename=WD14_MODEL_FILE, revision=WD14_MODEL_REVISION
        )
        tags_csv = hf_hub_download(
            WD14_MODEL_REPO, filename=WD14_TAGS_FILE, revision=WD14_MODEL_REVISION
        )
        self._session = onnxruntime.InferenceSession(
            weights, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._general_names, self._general_mask = parse_general_vocabulary(Path(tags_csv))

    def tag(self, image: np.ndarray) -> list[str]:  # pragma: no cover - needs session
        """Run WD14 and return the general tags above the threshold for a BGR ``image``."""
        self._load()
        feed = wd14_input(image, WD14_IMAGE_SIZE)
        input_name = self._session.get_inputs()[0].name
        probs = self._session.run(None, {input_name: feed})[0][0]
        return select_general_tags(probs, self._general_mask, self._general_names, self._threshold)


def make_tagger(settings: Settings) -> Tagger:
    """Return the tagger backend named by ``settings.caption_backend``.

    Only ``"wd14"`` is implemented; any other value raises so a typo or a
    not-yet-wired backend (e.g. JoyCaption/Florence-2) fails loudly.
    """
    backend = settings.caption_backend.strip().lower()
    if backend == "wd14":
        return Wd14Tagger(threshold=settings.caption_threshold)
    raise NotImplementedError(
        f"caption backend {backend!r} is not implemented; "
        f"set APP_CAPTION_BACKEND=wd14 (the only supported backend)."
    )


# --- Pure vocabulary / tag selection / caption building (no onnxruntime) ----


def parse_general_vocabulary(csv_path: Path) -> tuple[list[str], np.ndarray]:
    """Parse ``selected_tags.csv`` into (general tag names, boolean column mask).

    The CSV has columns ``tag_id,name,category,count``. The mask selects the
    category-0 (general/content) columns of the model's output; rating (9) and
    character (4) tags are excluded from a style dataset's content caption.
    """
    import csv

    import numpy as np

    names: list[str] = []
    is_general: list[bool] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            names.append(row["name"])
            is_general.append(int(row["category"]) == CATEGORY_GENERAL)
    mask = np.array(is_general, dtype=bool)
    general_names = [name for name, keep in zip(names, is_general, strict=True) if keep]
    return general_names, mask


def select_general_tags(
    probs: np.ndarray,
    general_mask: np.ndarray,
    general_names: list[str],
    threshold: float,
) -> list[str]:
    """Keep general tag names whose (already sigmoid'd) probability >= ``threshold``."""
    import numpy as np

    general_probs = np.asarray(probs)[general_mask]
    return [
        name
        for name, prob in zip(general_names, general_probs, strict=True)
        if float(prob) >= threshold
    ]


def build_caption(trigger: str, tags: list[str]) -> str:
    """Build a Style Locker caption: the trigger first, then content tags.

    The trigger is dropped from ``tags`` (case-insensitive) so it never repeats,
    and tags are de-duplicated while preserving order.
    """
    kept = [tag for tag in dict.fromkeys(tags) if tag.lower() != trigger.lower()]
    if not kept:
        return trigger
    return f"{trigger}, {', '.join(kept)}"


def wd14_input(image: np.ndarray, size: int) -> np.ndarray:
    """Build the NHWC ``[1, size, size, 3]`` float32 BGR feed from a BGR ``image``.

    Pads to a square with white (255), then resizes to ``size`` (AREA when
    downscaling, LANCZOS when upscaling). The model bakes mean/std normalization
    and the final sigmoid into the graph, so pixels stay in ``[0, 255]`` and the
    channel order stays BGR (the input is already ``cv2.imdecode``'d, so — unlike
    kohya which starts from PIL RGB — we do not swap channels).
    """
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    edge = max(height, width)
    pad_x, pad_y = edge - width, edge - height
    pad_l, pad_t = pad_x // 2, pad_y // 2
    padded = np.pad(
        image,
        ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)),
        mode="constant",
        constant_values=255,
    )
    interp = cv2.INTER_AREA if edge > size else cv2.INTER_LANCZOS4
    square = cv2.resize(padded, (size, size), interpolation=interp)
    return square.astype(np.float32)[np.newaxis, ...]


# --- I/O helpers -----------------------------------------------------------


def _decode_bgr(path: Path) -> np.ndarray | None:
    """Decode an image to a BGR array via a byte buffer (OS-agnostic, non-ASCII safe)."""
    import cv2
    import numpy as np

    buffer = np.fromfile(path, dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _write_png(image_bgr: np.ndarray, path: Path) -> None:
    """Save a BGR uint8 array as PNG via PIL (never ``cv2.imwrite`` on a path)."""
    from PIL import Image

    Image.fromarray(image_bgr[:, :, ::-1], mode="RGB").save(path)


def _write_caption(text: str, path: Path) -> None:
    """Write a caption to a UTF-8 ``.txt`` file (newline-terminated)."""
    path.write_text(f"{text}\n", encoding="utf-8")


# --- Orchestration ---------------------------------------------------------


def iter_images(clean_dir: Path) -> list[Path]:
    """Return clean image files under ``clean_dir`` in stable (sorted) order."""
    return image_files(clean_dir)


def caption_image(
    image_path: Path,
    *,
    tagger: Tagger,
    trigger: str,
    out_dir: Path,
) -> bool:
    """Caption one image into ``out_dir`` as ``<stem>.png`` + ``<stem>.txt``.

    Returns ``True`` when written, ``False`` when the image is unreadable.
    """
    image = _decode_bgr(image_path)
    if image is None:
        return False
    caption = build_caption(trigger, tagger.tag(image))
    stem = image_path.stem
    _write_png(image, out_dir / f"{stem}.png")
    _write_caption(caption, out_dir / f"{stem}.txt")
    return True


def _training_dir(ws: Workspace, settings: Settings) -> Path:
    """Resolve the kohya training folder ``05_dataset/<repeats>_<trigger>``."""
    return ws.training_dir(settings.dataset_repeats, settings.trigger_token)


def run(ctx: StageContext) -> StageResult:
    """Caption every image from ``04_clean`` into ``05_dataset/<N>_<trigger>``."""
    tag_component(COMPONENT)
    out = _training_dir(ctx.workspace, ctx.settings)
    out.mkdir(parents=True, exist_ok=True)

    tagger = make_tagger(ctx.settings)
    produced = 0
    for image_path in iter_images(ctx.workspace.clean):
        if caption_image(
            image_path, tagger=tagger, trigger=ctx.settings.trigger_token, out_dir=out
        ):
            produced += 1
    return StageResult(name=NAME, output_dir=out, produced=produced)


STAGE = Stage(
    name=NAME,
    summary=SUMMARY,
    component=COMPONENT,
    flag="run_caption",
    output=_training_dir,
    run=run,
)
