"""Application settings.

Single, typed source of runtime configuration. Reads from environment
variables and an optional local ``.env`` file. Secrets must never be
committed — keep them in ``.env`` (git-ignored) and document keys in
``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
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

    # --- VLM re-caption (Gemini prose via the proxy Space) ---
    hf_token: str = Field(
        default="",
        validation_alias=AliasChoices("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"),
        description="HF token (read access) for the private Gemini proxy Space. Not APP_-prefixed.",
    )
    vlm_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model for VLM re-captioning (agent default; the UI button uses pro).",
    )
    vlm_prompt_style: str = Field(
        default="rich",
        description="VLM caption style: 'rich' (more content, cleaner style residual) or 'optimal'.",
    )
    vlm_concurrency: int = Field(
        default=8, ge=1, description="Concurrent proxy calls when re-captioning a dataset."
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
    panel_resplit: bool = Field(
        default=True,
        description=(
            "Recursively X-Y-cut merged panel boxes along clean interior gutters, to "
            "recover touching / thin-gutter panels the contour pass lumped together. "
            "Coordinates-only (no resize). Disable if it over-splits busy artwork."
        ),
    )
    dedup_hamming_distance: int = Field(
        default=6,
        ge=0,
        description="Perceptual-hash distance below which two panels are near-duplicates.",
    )
    min_side_px: int = Field(
        default=256,
        ge=1,
        description=(
            "Floor below which a panel is too small to upscale cleanly and is routed "
            "to manual_review. Panels at or above it are kept and Lanczos-upscaled to "
            "target_side. Keep this well under target_side (a comic page yields many "
            "300-500px panels): a high floor silently dumps most of the dataset."
        ),
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
        default=10,
        ge=0,
        description=(
            "Dilate the bubble+text mask by this many px so the inpaint also eats the "
            "bubble outline and reaches the art around it (a small value leaves a white "
            "ring/empty bubble). Raise it if outlines survive; lower it if fills bleed."
        ),
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

    # --- Pipeline: Stage 4 (clean) ---
    clean_denoise: bool = Field(
        default=False,
        description="Apply a mild JPEG-artifact denoise pass (off by default: it softens "
        "the line texture a style LoRA should learn).",
    )

    # --- Pipeline: Stage 5 (caption) ---
    caption_backend: str = Field(
        default="wd14",
        description="Captioning backend. Only 'wd14' (WD14 ViT v3 ONNX) is implemented.",
    )
    caption_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Min WD14 confidence to keep a content tag (0.65-0.75 for 'reliable only').",
    )

    # --- Stage 6 (train) — local LoRA training via kohya sd-scripts ---
    train_model_type: str = Field(
        default="sd15",
        description="Base-model family to train: 'sd15', 'sdxl' or 'flux'. Selects the kohya "
        "sd-scripts entrypoint.",
    )
    train_base_model: str = Field(
        default="",
        description="Path to the base checkpoint (SD1.5/SDXL .safetensors, or the Flux "
        "transformer). Required to actually train; empty fails loudly at launch.",
    )
    train_sd_scripts_dir: Path = Field(
        default=Path("/home/serg/sd-scripts"),
        description="Local clone of kohya sd-scripts holding the *_train_network.py entrypoints.",
    )
    train_python: str = Field(
        default="",
        description="Python interpreter to launch sd-scripts with (needs a Blackwell-capable "
        "cu128 torch). Empty is resolved/validated by `doctor`.",
    )
    train_network_dim: int = Field(default=32, ge=1, description="LoRA network dimension (rank).")
    train_network_alpha: int = Field(
        default=16, ge=1, description="LoRA network alpha (scaling; commonly dim or dim/2)."
    )
    train_learning_rate: float = Field(default=1e-4, gt=0.0, description="Optimizer learning rate.")
    train_max_train_steps: int = Field(default=1600, ge=1, description="Total training steps.")
    train_resolution: int = Field(
        default=512,
        ge=64,
        description="Square training resolution (px): 512 for SD1.5, 1024 for SDXL/Flux.",
    )
    train_batch_size: int = Field(
        default=1, ge=1, description="Per-device train batch size (keep 1 on 16 GB VRAM)."
    )
    train_seed: int = Field(default=42, ge=0, description="Training RNG seed for reproducibility.")
    train_mixed_precision: str = Field(
        default="fp16",
        description="Mixed precision: 'no', 'fp16' or 'bf16' (bf16 recommended for SDXL/Flux).",
    )
    train_gradient_checkpointing: bool = Field(
        default=True,
        description="Trade compute for VRAM via gradient checkpointing (needed on 16 GB).",
    )
    train_cache_latents: bool = Field(
        default=True,
        description="Cache VAE latents to cut VRAM/time (disables crop/flip augmentation).",
    )
    train_output_name: str = Field(
        default="",
        description="Output LoRA filename stem (no extension). Empty uses the trigger token.",
    )
    train_optimizer_type: str = Field(
        default="AdamW8bit",
        description="kohya optimizer (AdamW8bit, AdamW, Lion, Prodigy, ...). AdamW8bit needs "
        "bitsandbytes.",
    )
    train_lr_scheduler: str = Field(
        default="constant",
        description="LR schedule (constant, cosine, cosine_with_restarts, ...).",
    )
    train_save_every_n_epochs: int = Field(
        default=1,
        ge=0,
        description="Save an intermediate LoRA every N epochs (0 disables intermediate saves).",
    )
    train_logging_dir: Path | None = Field(
        default=None,
        description="If set, write TensorBoard logs here (--logging_dir + --log_with=tensorboard).",
    )
    train_clip_skip: int = Field(
        default=2,
        ge=1,
        description="CLIP skip (SD 1.5 only; anime/comic convention is 2). Ignored on SDXL/Flux.",
    )
    train_unet_only: bool = Field(
        default=True,
        description="SDXL/Flux: train only the U-Net/DiT LoRA (VRAM-safe; text encoders left "
        "untrained). False also trains the text encoders (more VRAM, disables TE-output caching).",
    )
    train_text_encoder_lr: float | None = Field(
        default=None,
        description="SDXL: learning rate for both text encoders when train_unet_only is False. "
        "None falls back to the main learning_rate.",
    )
    train_flux_clip_l: str = Field(
        default="",
        description="Flux: path to the CLIP-L text-encoder .safetensors (required for Flux).",
    )
    train_flux_t5xxl: str = Field(
        default="",
        description="Flux: path to the T5-XXL text-encoder .safetensors (required for Flux).",
    )
    train_flux_ae: str = Field(
        default="",
        description="Flux: path to the autoencoder (VAE) .safetensors (required for Flux).",
    )
    train_flux_blocks_to_swap: int = Field(
        default=18,
        ge=0,
        le=35,
        description="Flux: transformer blocks swapped to CPU to fit VRAM (0 disables; 16 GB "
        "needs ~8-18, max 35).",
    )

    # --- Local web UI (S8) ---
    ui_host: str = Field(
        default="127.0.0.1",
        description="Host for the local UI server. Keep it loopback unless you know why.",
    )
    ui_port: int = Field(
        default=7860,
        ge=1,
        le=65535,
        description="Port for the local UI server.",
    )

    # --- Pipeline: stage flags (gate which stages 'run-all' executes) ---
    run_panels: bool = True
    run_bubbles: bool = True
    run_inpaint: bool = True
    run_clean: bool = True
    run_caption: bool = True
    # Training is heavy and optional: off by default so `run-all` stays the
    # dataset pipeline. `make-style-dataset train` runs it explicitly.
    run_train: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so the environment is parsed once. In tests, call
    ``get_settings.cache_clear()`` after mutating the environment.
    """
    return Settings()
