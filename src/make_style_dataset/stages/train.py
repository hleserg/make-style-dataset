"""Stage 6 — train a style LoRA from the kohya dataset via kohya sd-scripts.

Reads the kohya dataset folder ``05_dataset/<repeats>_<trigger>/`` and trains a
style LoRA into ``06_lora/<name>.safetensors`` by shelling out to a local clone
of kohya **sd-scripts**. The entrypoint is chosen by ``train_model_type``:
``train_network.py`` (SD 1.5), ``sdxl_train_network.py`` (SDXL) or
``flux_train_network.py`` (Flux). This module wires **SD 1.5** and **SDXL**
end-to-end; the Flux argument branch lands in a later subtask.

Pure-core-lazy-backend (shared with ``caption.py``/``inpaint.py``): the dataset
``--dataset_config`` TOML builder, the argument builder and the stdout progress
parser are pure and unit tested with a ``FakeTrainer``. Only the subprocess
launch (:class:`SdScriptsTrainer`) is uncovered. The heavy training runs in a
*separate* (root-owned) venv whose torch carries the Blackwell ``sm_120``
kernels — it is never imported into this package; we only spawn a process,
which keeps our ``numpy``/``Pillow`` pins isolated from sd-scripts'.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from make_style_dataset.media import image_files
from make_style_dataset.observability import tag_component
from make_style_dataset.stages.base import Stage, StageContext, StageResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from make_style_dataset.config import Settings
    from make_style_dataset.workspace import Workspace

NAME = "train"
SUMMARY = "Train a style LoRA from the kohya dataset via kohya sd-scripts."
COMPONENT = "stage:train"

#: ``train_model_type`` -> (entrypoint script, ``--network_module`` value).
ENTRYPOINTS: dict[str, tuple[str, str]] = {
    "sd15": ("train_network.py", "networks.lora"),
    "sdxl": ("sdxl_train_network.py", "networks.lora"),
    "flux": ("flux_train_network.py", "networks.lora_flux"),
}

#: Families with their ``build_train_args`` branch wired. flux lands in [T2].
SUPPORTED_MODEL_TYPES = frozenset({"sd15", "sdxl"})

#: ``epoch 1/10`` banner (kohya prints it on stdout between epochs).
_EPOCH_RE = re.compile(r"^epoch (\d+)/(\d+)")
#: ``... | 200/2000 [01:30<13:30, ...`` — the trailing ``[`` excludes the
#: latent/text-encoder caching bar, which prints ``N/total`` without a time bracket.
_STEP_RE = re.compile(r"(\d+)/(\d+)\s*\[")
#: ``avr_loss=0.0987`` (also matches scientific notation like ``1e-05``).
_LOSS_RE = re.compile(r"avr_loss=([0-9.eE+\-]+)")


@dataclass(frozen=True)
class TrainProgress:
    """A parsed line of sd-scripts training progress (step bar or epoch banner)."""

    step: int | None = None
    total_steps: int | None = None
    epoch: int | None = None
    total_epochs: int | None = None
    loss: float | None = None

    @property
    def fraction(self) -> float | None:
        """Completed fraction in ``[0, 1]`` from the step counter, or ``None``."""
        if self.step is None or not self.total_steps:
            return None
        return self.step / self.total_steps


@dataclass(frozen=True)
class TrainPlan:
    """A fully-resolved training invocation: what to run, where, what it produces."""

    command: list[str]
    cwd: Path
    output_path: Path


class Trainer(Protocol):
    """Runs a :class:`TrainPlan` to completion and returns the produced LoRA path."""

    def train(self, plan: TrainPlan) -> Path:
        """Run training; return the produced ``.safetensors``. Raise on failure."""
        ...


# PLAYBOOK-START
# id: pure-core-lazy-backend
# title: Pure policy core behind a lazily-imported heavy backend
# status: draft
# category: testability
# tags: [testing, dependency-injection, coverage, subprocess]
# Same split as caption.py/inpaint.py, here for an *external process* backend:
# the dataset-TOML builder, the argv builder and the stdout progress parser are
# pure and unit tested; only the subprocess launch stays uncovered. The heavy
# trainer lives in a separate venv we shell out to — never imported — so its
# conflicting deps can't reach our environment.
# PLAYBOOK-END
class SdScriptsTrainer:
    """Launch kohya sd-scripts as a subprocess in its own (cu128/sm_120) venv."""

    def __init__(self, on_progress: Callable[[TrainProgress], None] | None = None) -> None:
        self._on_progress = on_progress

    def train(self, plan: TrainPlan) -> Path:  # pragma: no cover - spawns a GPU subprocess
        """Run the plan, stream + parse progress, verify the LoRA was written."""
        import os
        import shlex
        import subprocess  # nosec B404 - trusted, list-form, shell=False command

        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        process = subprocess.Popen(  # nosec B603 - fixed argv from validated settings, no shell
            plan.command,
            cwd=plan.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # the tqdm bar is on stderr; merge it in
            text=True,
            bufsize=1,
            env=env,
        )
        stdout = process.stdout
        if stdout is None:  # PIPE always yields a stream; satisfy the type checker
            raise RuntimeError("failed to capture sd-scripts output")
        saved = False
        for line in stdout:
            stripped = line.rstrip()
            print(stripped)
            if "model saved." in stripped:
                saved = True
            progress = parse_progress(stripped)
            if progress is not None and self._on_progress is not None:
                self._on_progress(progress)
        code = process.wait()
        if code != 0:
            raise RuntimeError(
                f"sd-scripts training failed (exit {code}): {shlex.join(plan.command)}"
            )
        if not (saved and plan.output_path.is_file() and plan.output_path.stat().st_size > 0):
            raise RuntimeError(
                f"training finished but {plan.output_path} is missing or empty (see the log above)."
            )
        return plan.output_path


def make_trainer(
    settings: Settings, on_progress: Callable[[TrainProgress], None] | None = None
) -> Trainer:
    """Return the training backend (only the sd-scripts subprocess backend exists)."""
    del settings  # backend is the same for every family; the family changes the command
    return SdScriptsTrainer(on_progress=on_progress)


# --- Pure: dataset TOML, argv, progress parsing (no torch, no subprocess) ---


def parse_progress(line: str) -> TrainProgress | None:
    """Parse one sd-scripts stdout/stderr line into :class:`TrainProgress`, or ``None``.

    Recognises the ``epoch N/M`` banner and the tqdm training bar
    (``... 200/2000 [01:30<13:30, ... avr_loss=0.0987]``). The step match is
    anchored on ``avr_loss=`` so the latent/text-encoder caching bar — which
    also prints ``N/total`` but without a loss — never false-matches.
    """
    epoch_match = _EPOCH_RE.match(line)
    if epoch_match:
        return TrainProgress(
            epoch=int(epoch_match.group(1)), total_epochs=int(epoch_match.group(2))
        )
    if "avr_loss=" not in line:
        return None
    step_match = _STEP_RE.search(line)
    if step_match is None:
        return None
    loss_match = _LOSS_RE.search(line)
    return TrainProgress(
        step=int(step_match.group(1)),
        total_steps=int(step_match.group(2)),
        loss=float(loss_match.group(1)) if loss_match else None,
    )


def _max_bucket_reso(model_type: str) -> int:
    """Largest bucket edge per family (sd15 trains at 512, SDXL/Flux at 1024)."""
    return 1024 if model_type == "sd15" else 1536


def build_dataset_toml(
    *,
    image_dir: Path,
    num_repeats: int,
    resolution: int,
    batch_size: int,
    max_bucket_reso: int,
    bucket_reso_steps: int = 64,
) -> str:
    """Render the kohya ``--dataset_config`` TOML for one style-LoRA dataset.

    ``caption_extension='.txt'`` is mandatory: kohya's argparse default is
    ``.caption``, so without it our sidecars are silently ignored. ``keep_tokens
    =1`` + ``shuffle_caption=false`` pin the leading trigger token that the
    caption stage writes first in every ``.txt``. ``num_repeats`` is explicit
    because kohya does not parse the ``<N>_<trigger>`` folder name under
    ``--dataset_config``.
    """
    image_path = image_dir.resolve().as_posix()
    return (
        "[general]\n"
        "caption_extension = '.txt'\n"
        "shuffle_caption = false\n"
        "keep_tokens = 1\n"
        "\n"
        "[[datasets]]\n"
        f"resolution = {resolution}\n"
        f"batch_size = {batch_size}\n"
        "enable_bucket = true\n"
        "min_bucket_reso = 256\n"
        f"max_bucket_reso = {max_bucket_reso}\n"
        "bucket_no_upscale = true\n"
        f"bucket_reso_steps = {bucket_reso_steps}\n"
        "\n"
        "  [[datasets.subsets]]\n"
        f"  image_dir = '{image_path}'\n"
        f"  num_repeats = {num_repeats}\n"
    )


def build_train_args(
    settings: Settings,
    *,
    dataset_config: Path,
    output_dir: Path,
    output_name: str,
) -> list[str]:
    """Build the sd-scripts script arguments (appended after the entrypoint).

    Covers the flags common to every family plus the SD 1.5 and SDXL specifics.
    Raises :class:`NotImplementedError` for flux (wired in [T2]) and
    :class:`ValueError` when the base checkpoint is unset.
    """
    model_type = settings.train_model_type.strip().lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise NotImplementedError(
            f"train_model_type={model_type!r} is not wired yet; "
            f"set APP_TRAIN_MODEL_TYPE to sd15 or sdxl (flux lands in [T2])."
        )
    if not settings.train_base_model:
        raise ValueError(
            "train_base_model is empty; set APP_TRAIN_BASE_MODEL to a base checkpoint path."
        )
    _, network_module = ENTRYPOINTS[model_type]
    args = [
        f"--pretrained_model_name_or_path={settings.train_base_model}",
        f"--network_module={network_module}",
        f"--output_dir={output_dir.resolve().as_posix()}",
        f"--output_name={output_name}",
        f"--dataset_config={dataset_config.resolve().as_posix()}",
        f"--max_train_steps={settings.train_max_train_steps}",
        f"--network_dim={settings.train_network_dim}",
        f"--network_alpha={settings.train_network_alpha}",
        f"--learning_rate={settings.train_learning_rate}",
        f"--train_batch_size={settings.train_batch_size}",
        f"--resolution={settings.train_resolution}",
        f"--mixed_precision={settings.train_mixed_precision}",
        f"--seed={settings.train_seed}",
        f"--optimizer_type={settings.train_optimizer_type}",
        f"--lr_scheduler={settings.train_lr_scheduler}",
        "--save_model_as=safetensors",
        "--sdpa",
    ]
    if settings.train_gradient_checkpointing:
        args.append("--gradient_checkpointing")
    if settings.train_cache_latents:
        args.append("--cache_latents")
        args.append("--cache_latents_to_disk")
    if settings.train_save_every_n_epochs > 0:
        args.append(f"--save_every_n_epochs={settings.train_save_every_n_epochs}")
    if settings.train_logging_dir is not None:
        args.append(f"--logging_dir={settings.train_logging_dir.resolve().as_posix()}")
        args.append("--log_with=tensorboard")
    if model_type == "sd15":
        args.append(f"--clip_skip={settings.train_clip_skip}")
    elif model_type == "sdxl":
        # SDXL VAE produces NaN/black latents in fp16/bf16 -> always keep it fp32.
        args.append("--no_half_vae")
        if settings.train_unet_only:
            # Unet-only fits 16 GB; text-encoder-output caching REQUIRES it
            # (kohya asserts the pair) and needs shuffle_caption off (our TOML sets it).
            args.append("--network_train_unet_only")
            args.append("--cache_text_encoder_outputs")
            args.append("--cache_text_encoder_outputs_to_disk")
        elif settings.train_text_encoder_lr is not None:
            # Train both text encoders; --text_encoder_lr is nargs=* (two values).
            lr = str(settings.train_text_encoder_lr)
            args.extend(["--text_encoder_lr", lr, lr])
    return args


def resolve_python(settings: Settings) -> str:
    """Interpreter to launch sd-scripts with — its own cu128/sm_120 venv by default."""
    if settings.train_python:
        return settings.train_python
    return (settings.train_sd_scripts_dir / "venv" / "bin" / "python").as_posix()


def build_launch_command(settings: Settings, script_args: list[str]) -> list[str]:
    """Build the full argv: venv python -> ``accelerate launch`` -> entrypoint -> args.

    Uses the explicit ``-m accelerate.commands.launch`` module form (this box has
    no accelerate default config, so a bare ``accelerate launch`` would prompt).
    """
    model_type = settings.train_model_type.strip().lower()
    if model_type not in ENTRYPOINTS:
        raise NotImplementedError(f"unknown train_model_type={model_type!r}")
    entrypoint, _ = ENTRYPOINTS[model_type]
    return [
        resolve_python(settings),
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "--num_cpu_threads_per_process",
        "1",
        "--mixed_precision",
        settings.train_mixed_precision,
        "--dynamo_backend",
        "no",
        entrypoint,
        *script_args,
    ]


def resolve_output_name(settings: Settings) -> str:
    """LoRA filename stem: ``train_output_name`` if set, else the trigger token."""
    return settings.train_output_name.strip() or settings.trigger_token


# --- Orchestration ---------------------------------------------------------


def _lora_dir(ws: Workspace, settings: Settings) -> Path:
    """Resolve the stage's output directory (``06_lora``)."""
    del settings
    return ws.lora


def build_plan(
    settings: Settings, *, dataset_dir: Path, output_dir: Path, dataset_config: Path
) -> TrainPlan:
    """Assemble the full :class:`TrainPlan` from settings and resolved paths (pure)."""
    del dataset_dir  # paths already resolved by the caller; kept for call-site clarity
    output_name = resolve_output_name(settings)
    script_args = build_train_args(
        settings, dataset_config=dataset_config, output_dir=output_dir, output_name=output_name
    )
    command = build_launch_command(settings, script_args)
    return TrainPlan(
        command=command,
        cwd=settings.train_sd_scripts_dir,
        output_path=output_dir / f"{output_name}.safetensors",
    )


def run(ctx: StageContext) -> StageResult:
    """Train a style LoRA from ``05_dataset/<N>_<trigger>`` into ``06_lora/``."""
    tag_component(COMPONENT)
    settings = ctx.settings
    out = _lora_dir(ctx.workspace, settings)
    out.mkdir(parents=True, exist_ok=True)

    dataset_dir = ctx.workspace.training_dir(settings.dataset_repeats, settings.trigger_token)
    if not image_files(dataset_dir):
        raise ValueError(
            f"no dataset images in {dataset_dir}; run the caption stage (or the full "
            "pipeline) first to produce the kohya dataset."
        )

    model_type = settings.train_model_type.strip().lower()
    toml_text = build_dataset_toml(
        image_dir=dataset_dir,
        num_repeats=settings.dataset_repeats,
        resolution=settings.train_resolution,
        batch_size=settings.train_batch_size,
        max_bucket_reso=_max_bucket_reso(model_type),
    )
    dataset_config = out / "dataset.toml"
    dataset_config.write_text(toml_text, encoding="utf-8")

    plan = build_plan(
        settings, dataset_dir=dataset_dir, output_dir=out, dataset_config=dataset_config
    )
    trainer = make_trainer(settings)
    trainer.train(plan)  # returns the verified .safetensors path or raises
    return StageResult(name=NAME, output_dir=out, produced=1)


STAGE = Stage(
    name=NAME,
    summary=SUMMARY,
    component=COMPONENT,
    flag="run_train",
    output=_lora_dir,
    run=run,
)
