"""First-run helpers: scaffold the workspace and diagnose the environment.

Two operator-facing conveniences exposed as the ``init`` and ``doctor`` CLI
subcommands. The goal is to remove the fiddly manual steps a non-technical user
would otherwise face: creating folders, copying ``.env``, and figuring out
whether the GPU stack is actually usable.

The logic here is pure and testable (filesystem effects go through an injected
:class:`~make_style_dataset.workspace.Workspace` and explicit paths); the heavy
environment probes — importing ``torch`` / ``onnxruntime`` — are isolated behind
small functions so the rest stays covered without the optional ``gpu`` group
installed.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from make_style_dataset.media import image_files
from make_style_dataset.workspace import Workspace

if TYPE_CHECKING:
    from make_style_dataset.config import Settings

#: Default location of the committed template env file (relative to the cwd).
DEFAULT_ENV_EXAMPLE = Path(".env.example")
#: Default location of the user's local env file (git-ignored).
DEFAULT_ENV_PATH = Path(".env")


# --- init -----------------------------------------------------------------


@dataclass(frozen=True)
class InitOutcome:
    """What :func:`initialize_workspace` did, so the report can describe it."""

    created_dirs: list[Path] = field(default_factory=list)
    env_created: bool = False
    env_path: Path = DEFAULT_ENV_PATH
    env_example_missing: bool = False


def initialize_workspace(
    workspace: Workspace,
    *,
    env_example: Path = DEFAULT_ENV_EXAMPLE,
    env_path: Path = DEFAULT_ENV_PATH,
) -> InitOutcome:
    """Create the input/review folders and seed ``.env`` from the template.

    Idempotent: existing folders are left alone and an existing ``.env`` is never
    overwritten (it may hold the user's settings). Returns an :class:`InitOutcome`
    describing exactly what changed.
    """
    created: list[Path] = []
    for path in (workspace.root, workspace.pages, workspace.manual_review):
        if not path.exists():
            created.append(path)
        path.mkdir(parents=True, exist_ok=True)

    env_created = False
    env_example_missing = False
    if env_path.exists():
        env_created = False
    elif env_example.exists():
        shutil.copyfile(env_example, env_path)
        env_created = True
    else:
        env_example_missing = True

    return InitOutcome(
        created_dirs=created,
        env_created=env_created,
        env_path=env_path,
        env_example_missing=env_example_missing,
    )


def format_init_report(outcome: InitOutcome, workspace: Workspace) -> str:
    """Render a friendly, copy-pasteable summary of an :func:`initialize_workspace`."""
    lines = ["Workspace ready."]
    if outcome.created_dirs:
        lines.append("  Created folders:")
        lines += [f"    {path}" for path in outcome.created_dirs]
    else:
        lines.append("  Folders already existed (nothing to create).")

    if outcome.env_created:
        lines.append(f"  Created {outcome.env_path} from .env.example — edit it to taste.")
    elif outcome.env_example_missing:
        lines.append("  No .env.example found; skipped .env (using built-in defaults).")
    else:
        lines.append(f"  {outcome.env_path} already exists — left untouched.")

    lines += [
        "",
        "Next steps:",
        f"  1. Drop your comic pages into:  {workspace.pages}",
        "  2. Check your machine is ready:  make-style-dataset doctor",
        "  3. Build the dataset:            make-style-dataset run-all",
    ]
    return "\n".join(lines)


# --- doctor ----------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One environment diagnostic line: a name, a pass/fail, and a detail."""

    name: str
    ok: bool
    detail: str


#: A GPU probe is a zero-arg function returning a single :class:`Check`. Injecting
#: them keeps the optional-import logic out of the covered code path.
GpuProbe = Callable[[], Check]


def probe_python() -> Check:
    """Report the running interpreter version (>= 3.12 is required)."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 12)
    return Check("python", ok, version if ok else f"{version} (need >= 3.12)")


def probe_venv() -> Check:
    """Report whether we're running inside a virtual environment."""
    in_venv = sys.prefix != sys.base_prefix
    return Check("venv", in_venv, "active" if in_venv else "not in a venv (use `uv run`)")


def probe_torch() -> Check:  # pragma: no cover - exercises the optional gpu group
    """Report torch + CUDA availability (the bubbles stage runs on torch)."""
    try:
        import torch  # pyright: ignore[reportMissingImports]
    except ImportError:
        return Check("torch / CUDA", False, "torch not installed (run: uv sync --group gpu)")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        return Check("torch / CUDA", True, f"{torch.__version__}, GPU: {gpu}")
    return Check("torch / CUDA", False, f"{torch.__version__}, no CUDA GPU (CPU fallback, slow)")


def probe_onnxruntime() -> Check:  # pragma: no cover - exercises the optional gpu group
    """Report onnxruntime providers (inpaint + caption run on ONNX)."""
    try:
        import onnxruntime  # pyright: ignore[reportMissingImports]
    except ImportError:
        return Check("onnxruntime", False, "not installed (run: uv sync --group gpu)")
    providers = onnxruntime.get_available_providers()
    cuda = "CUDAExecutionProvider" in providers
    return Check("onnxruntime", cuda, f"providers: {', '.join(providers)}")


def probe_workspace(workspace: Workspace) -> list[Check]:
    """Report whether the input folder exists and how many pages are waiting."""
    pages_dir = workspace.pages
    if not pages_dir.is_dir():
        return [Check("workspace", False, f"{pages_dir} missing (run: make-style-dataset init)")]
    page_count = len(image_files(pages_dir))
    detail = f"{pages_dir}: {page_count} page(s) ready"
    return [Check("workspace", True, detail), Check("pages", page_count > 0, f"{page_count} found")]


def probe_env(env_path: Path) -> Check:
    """Report whether a local ``.env`` exists (defaults are used otherwise)."""
    exists = env_path.exists()
    return Check("config (.env)", True, "present" if exists else "absent (using built-in defaults)")


# --- training-env probes (stage 6) ----------------------------------------

#: A training-interpreter probe takes Settings and returns one Check. Injected +
#: ``pragma: no cover`` like the GPU probes: it spawns the *separate* trainer venv.
TrainProbe = Callable[["Settings"], Check]


def _probe_model_path(name: str, value: str, env: str) -> Check:
    """A Check for a model-file setting: set, and present on disk."""
    if not value:
        return Check(name, False, f"{env} not set")
    exists = Path(value).exists()
    return Check(name, exists, value if exists else f"{value} not found")


def probe_train_paths(settings: Settings) -> list[Check]:
    """Check the sd-scripts clone, its entrypoint, and the base/Flux model paths."""
    from make_style_dataset.stages.train import ENTRYPOINTS

    checks: list[Check] = []
    model_type = settings.train_model_type.strip().lower()
    scripts_dir = settings.train_sd_scripts_dir
    entry = ENTRYPOINTS.get(model_type)
    if not scripts_dir.is_dir():
        checks.append(
            Check("sd-scripts", False, f"{scripts_dir} missing (clone kohya sd-scripts there)")
        )
    elif entry is None:
        checks.append(Check("sd-scripts", False, f"unknown APP_TRAIN_MODEL_TYPE={model_type!r}"))
    else:
        entrypoint = scripts_dir / entry[0]
        ok = entrypoint.is_file()
        checks.append(Check("sd-scripts", ok, str(entrypoint) if ok else f"{entrypoint} missing"))

    checks.append(
        _probe_model_path("base model", settings.train_base_model, "APP_TRAIN_BASE_MODEL")
    )
    if model_type == "flux":
        checks.append(
            _probe_model_path("flux clip_l", settings.train_flux_clip_l, "APP_TRAIN_FLUX_CLIP_L")
        )
        checks.append(
            _probe_model_path("flux t5xxl", settings.train_flux_t5xxl, "APP_TRAIN_FLUX_T5XXL")
        )
        checks.append(_probe_model_path("flux ae", settings.train_flux_ae, "APP_TRAIN_FLUX_AE"))
    return checks


def probe_train_python(settings: Settings) -> Check:  # pragma: no cover - spawns the trainer venv
    """Check the trainer interpreter carries a Blackwell-capable (sm_120) torch."""
    import subprocess  # nosec B404 - fixed list-form command, shell=False

    from make_style_dataset.stages.train import resolve_python

    python = resolve_python(settings)
    code = "import torch; print(';'.join(torch.cuda.get_arch_list()))"
    try:
        result = subprocess.run(  # nosec B603 - interpreter path from settings, no shell
            [python, "-c", code], capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Check("trainer torch", False, f"{python}: {exc}")
    if result.returncode != 0:
        return Check("trainer torch", False, f"{python}: torch import failed")
    archs = result.stdout.strip()
    ok = "sm_120" in archs.split(";")
    detail = f"arch_list={archs}" if ok else f"arch_list={archs} (no sm_120 — wrong torch build!)"
    return Check("trainer torch", ok, detail)


def gather_checks(
    workspace: Workspace,
    *,
    env_path: Path = DEFAULT_ENV_PATH,
    gpu_probes: Sequence[GpuProbe] = (probe_torch, probe_onnxruntime),
    settings: Settings | None = None,
    train_python_probe: TrainProbe = probe_train_python,
) -> list[Check]:
    """Collect every diagnostic into an ordered list of :class:`Check`.

    When ``settings`` is given, the stage-6 training-env checks are appended (the
    sd-scripts clone, model paths, and the trainer interpreter's ``sm_120`` torch).
    """
    checks = [probe_python(), probe_venv(), probe_env(env_path)]
    checks += probe_workspace(workspace)
    checks += [probe() for probe in gpu_probes]
    if settings is not None:
        checks += probe_train_paths(settings)
        checks.append(train_python_probe(settings))
    return checks


def format_doctor_report(checks: Iterable[Check]) -> str:
    """Render checks as an aligned ``[ok]``/``[--]`` table with a verdict."""
    checks = list(checks)
    width = max((len(check.name) for check in checks), default=0)
    lines = ["Environment check:"]
    for check in checks:
        mark = "ok" if check.ok else "--"
        lines.append(f"  [{mark}] {check.name:<{width}}  {check.detail}")
    if all(check.ok for check in checks):
        lines.append("\nAll good — you're ready to run the pipeline.")
    else:
        lines.append("\nSome checks need attention (see the lines marked [--] above).")
    return "\n".join(lines)
