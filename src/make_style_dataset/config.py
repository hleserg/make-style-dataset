"""Application settings.

Single, typed source of runtime configuration. Reads from environment
variables and an optional local ``.env`` file. Secrets must never be
committed — keep them in ``.env`` (git-ignored) and document keys in
``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from the environment.

    # PLAYBOOK-START
    # id: typed-settings-singleton
    # title: Typed settings as a cached singleton
    # status: refined
    # category: configuration
    # tags: [pydantic, config, 12factor]
    # Centralize all env access in one typed object resolved once via an
    # lru_cache'd accessor. Code never reads os.environ directly; tests
    # override by clearing the cache. Substitution test passes: useful in
    # any 12-factor service regardless of domain.
    # PLAYBOOK-END
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        extra="ignore",
    )

    environment: str = "development"
    debug: bool = False

    sentry_dsn: str = ""
    sentry_environment: str = "development"
    sentry_release: str = ""
    sentry_traces_sample_rate: float = 0.0

    # --- Pipeline: workspace + dataset identity ---
    workspace: Path = Field(
        default=Path("workspace"),
        description="Root directory holding all intermediate and output stage folders.",
    )
    trigger_token: str = Field(
        default="comicstyle",
        description="LoRA trigger word; also names the kohya training folder.",
    )
    dataset_repeats: int = Field(
        default=10,
        ge=1,
        description="kohya repeat count; the dataset folder is named '<repeats>_<trigger>'.",
    )

    # --- Pipeline: stage thresholds ---
    min_panel_area: int = Field(
        default=10_000,
        ge=0,
        description="Discard detected panels smaller than this area (px^2).",
    )
    panel_border: int = Field(
        default=8,
        ge=0,
        description="Pixels trimmed off each side of a detected panel to drop frame/gutters.",
    )
    max_panels: int = Field(
        default=12,
        ge=1,
        description="Pages with more panels than this go to manual_review (bad segmentation).",
    )
    splash_area_ratio: float = Field(
        default=0.85,
        gt=0.0,
        le=1.0,
        description="A lone panel covering at least this fraction of the page is a splash.",
    )
    dedup_hamming_distance: int = Field(
        default=6,
        ge=0,
        description="Perceptual-hash distance below which two panels are near-duplicates.",
    )
    min_side_px: int = Field(
        default=512,
        ge=1,
        description="Drop panels whose shorter side is below this many pixels.",
    )
    target_side: int = Field(
        default=1024,
        ge=1,
        description="Target shorter-side length (px) for dataset images.",
    )

    # --- Pipeline: Stage 2 (bubbles) detection + masking ---
    bubble_model: str = Field(
        default="kitsumed/yolov8m_seg-speech-bubble",
        description="YOLOv8-seg weights: a Hugging Face repo id or a local .pt path.",
    )
    bubble_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum YOLOv8 confidence for a speech-bubble detection to count.",
    )
    ocr_languages: str = Field(
        default="en",
        description="Comma-separated EasyOCR language codes for SFX/text detection (e.g. 'en,ja').",
    )
    mask_dilation_px: int = Field(
        default=5,
        ge=0,
        description="Dilate the bubble+text mask by this many px to catch outlines/letter strokes.",
    )
    max_mask_coverage: float = Field(
        default=0.6,
        gt=0.0,
        le=1.0,
        description="Panels whose mask covers more than this fraction go to manual_review.",
    )
    bubbles_debug: bool = Field(
        default=False,
        description="Also write a mask/panel overlay to manual_review for visual inspection.",
    )

    # --- Pipeline: Stage 3 (inpaint) ---
    inpaint_backend: str = Field(
        default="lama",
        description="Inpainting backend. Only 'lama' (ONNX Big-LaMa) is implemented.",
    )

    # --- Pipeline: stage flags (gate which stages 'run-all' executes) ---
    run_panels: bool = True
    run_bubbles: bool = True
    run_inpaint: bool = True
    run_clean: bool = True
    run_caption: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so the environment is parsed once. In tests, call
    ``get_settings.cache_clear()`` after mutating the environment.
    """
    return Settings()
