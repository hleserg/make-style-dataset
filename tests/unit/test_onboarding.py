"""Tests for make_style_dataset.onboarding (init + doctor logic)."""

from __future__ import annotations

from pathlib import Path

from make_style_dataset.config import Settings
from make_style_dataset.onboarding import (
    Check,
    format_doctor_report,
    format_init_report,
    gather_checks,
    initialize_workspace,
    probe_env,
    probe_python,
    probe_train_paths,
    probe_venv,
    probe_workspace,
)
from make_style_dataset.workspace import Workspace


def _ws(tmp_path: Path) -> Workspace:
    return Workspace(root=tmp_path / "ws")


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {"workspace": tmp_path / "ws"}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _sd_scripts(tmp_path: Path, entrypoint: str = "train_network.py") -> Path:
    """A fake sd-scripts clone containing one entrypoint."""
    scripts = tmp_path / "sd-scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / entrypoint).write_text("x", encoding="utf-8")
    return scripts


# --- init ------------------------------------------------------------------


def test_initialize_creates_dirs_and_env(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    example = tmp_path / ".env.example"
    example.write_text("APP_TRIGGER_TOKEN=x\n", encoding="utf-8")
    env = tmp_path / ".env"

    outcome = initialize_workspace(ws, env_example=example, env_path=env)

    assert ws.pages.is_dir()
    assert ws.manual_review.is_dir()
    assert ws.root in outcome.created_dirs
    assert outcome.env_created is True
    assert env.read_text(encoding="utf-8") == "APP_TRIGGER_TOKEN=x\n"


def test_initialize_is_idempotent_and_keeps_existing_env(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    example = tmp_path / ".env.example"
    example.write_text("template\n", encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text("user-edited\n", encoding="utf-8")

    outcome = initialize_workspace(ws, env_example=example, env_path=env)

    assert outcome.created_dirs  # first run still creates the folders
    assert outcome.env_created is False
    assert env.read_text(encoding="utf-8") == "user-edited\n"  # never clobbered

    again = initialize_workspace(ws, env_example=example, env_path=env)
    assert again.created_dirs == []  # nothing left to create


def test_initialize_handles_missing_example(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    outcome = initialize_workspace(
        ws, env_example=tmp_path / "nope.example", env_path=tmp_path / ".env"
    )
    assert outcome.env_created is False
    assert outcome.env_example_missing is True


def test_format_init_report_created(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    example = tmp_path / ".env.example"
    example.write_text("x\n", encoding="utf-8")
    outcome = initialize_workspace(ws, env_example=example, env_path=tmp_path / ".env")

    report = format_init_report(outcome, ws)
    assert "Created folders:" in report
    assert "from .env.example" in report
    assert str(ws.pages) in report
    assert "make-style-dataset run-all" in report


def test_format_init_report_nothing_new(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    env = tmp_path / ".env"
    env.write_text("x\n", encoding="utf-8")
    initialize_workspace(ws, env_example=tmp_path / "missing", env_path=env)
    # Second run: dirs exist, env exists -> both "nothing changed" branches.
    outcome = initialize_workspace(ws, env_example=tmp_path / "missing", env_path=env)

    report = format_init_report(outcome, ws)
    assert "already existed" in report
    assert "left untouched" in report


def test_format_init_report_example_missing(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    outcome = initialize_workspace(ws, env_example=tmp_path / "missing", env_path=tmp_path / ".env")
    report = format_init_report(outcome, ws)
    assert "No .env.example found" in report


# --- doctor probes ---------------------------------------------------------


def test_probe_python_and_venv_and_env(tmp_path: Path) -> None:
    py = probe_python()
    assert py.name == "python" and py.ok is True  # test interpreter is >= 3.12

    venv = probe_venv()
    assert venv.name == "venv"  # ok depends on runner; just shape

    present = probe_env(tmp_path / ".env")
    assert present.ok is True and "absent" in present.detail
    (tmp_path / ".env").write_text("x", encoding="utf-8")
    assert "present" in probe_env(tmp_path / ".env").detail


def test_probe_workspace_missing(tmp_path: Path) -> None:
    checks = probe_workspace(_ws(tmp_path))
    assert len(checks) == 1
    assert checks[0].ok is False
    assert "init" in checks[0].detail


def test_probe_workspace_counts_pages(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    ws.pages.mkdir(parents=True)
    (ws.pages / "a.png").write_bytes(b"x")
    (ws.pages / "notes.txt").write_text("x", encoding="utf-8")  # not an image
    (ws.pages / "sub").mkdir()  # not a file
    checks = probe_workspace(ws)
    names = {c.name: c for c in checks}
    assert names["workspace"].ok is True
    assert "1 page" in names["workspace"].detail
    assert names["pages"].ok is True


def test_probe_workspace_zero_pages(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    ws.pages.mkdir(parents=True)
    pages_check = next(c for c in probe_workspace(ws) if c.name == "pages")
    assert pages_check.ok is False


# --- doctor assembly + report ---------------------------------------------


def test_gather_checks_uses_injected_gpu_probes(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    ws.pages.mkdir(parents=True)
    fake = [
        lambda: Check("torch / CUDA", True, "fake-cuda"),
        lambda: Check("onnxruntime", True, "fake-ort"),
    ]
    checks = gather_checks(ws, env_path=tmp_path / ".env", gpu_probes=fake)
    names = [c.name for c in checks]
    assert names == [
        "python",
        "venv",
        "config (.env)",
        "workspace",
        "pages",
        "torch / CUDA",
        "onnxruntime",
    ]


# --- training-env probes (stage 6) ----------------------------------------


def test_probe_train_paths_missing_sd_scripts(tmp_path: Path) -> None:
    settings = _settings(tmp_path, train_sd_scripts_dir=tmp_path / "absent")
    checks = {c.name: c for c in probe_train_paths(settings)}
    assert checks["sd-scripts"].ok is False
    assert "missing" in checks["sd-scripts"].detail


def test_probe_train_paths_ok_sd15(tmp_path: Path) -> None:
    scripts = _sd_scripts(tmp_path)
    model = tmp_path / "base.safetensors"
    model.write_bytes(b"x")
    settings = _settings(tmp_path, train_sd_scripts_dir=scripts, train_base_model=str(model))
    checks = {c.name: c for c in probe_train_paths(settings)}
    assert checks["sd-scripts"].ok is True
    assert checks["base model"].ok is True


def test_probe_train_paths_entrypoint_missing(tmp_path: Path) -> None:
    scripts = tmp_path / "sd-scripts"
    scripts.mkdir()  # dir exists but has no train_network.py
    settings = _settings(tmp_path, train_sd_scripts_dir=scripts)
    sd = next(c for c in probe_train_paths(settings) if c.name == "sd-scripts")
    assert sd.ok is False and "missing" in sd.detail


def test_probe_train_paths_unknown_model_type(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path, train_sd_scripts_dir=_sd_scripts(tmp_path), train_model_type="sd3"
    )
    sd = next(c for c in probe_train_paths(settings) if c.name == "sd-scripts")
    assert sd.ok is False and "unknown" in sd.detail.lower()


def test_probe_train_paths_base_model_unset_and_missing(tmp_path: Path) -> None:
    scripts = _sd_scripts(tmp_path)
    base_unset = next(
        c
        for c in probe_train_paths(_settings(tmp_path, train_sd_scripts_dir=scripts))
        if c.name == "base model"
    )
    assert base_unset.ok is False and "not set" in base_unset.detail
    ghost = _settings(
        tmp_path, train_sd_scripts_dir=scripts, train_base_model=str(tmp_path / "ghost.safetensors")
    )
    base_missing = next(c for c in probe_train_paths(ghost) if c.name == "base model")
    assert base_missing.ok is False and "not found" in base_missing.detail


def test_probe_train_paths_flux_components(tmp_path: Path) -> None:
    scripts = _sd_scripts(tmp_path, entrypoint="flux_train_network.py")
    clip = tmp_path / "clip_l.safetensors"
    clip.write_bytes(b"x")
    settings = _settings(
        tmp_path,
        train_sd_scripts_dir=scripts,
        train_model_type="flux",
        train_flux_clip_l=str(clip),
    )
    checks = {c.name: c for c in probe_train_paths(settings)}
    assert checks["flux clip_l"].ok is True
    assert checks["flux t5xxl"].ok is False  # not set
    assert checks["flux ae"].ok is False


def test_gather_checks_appends_train_when_settings(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    ws.pages.mkdir(parents=True)
    settings = _settings(tmp_path, train_sd_scripts_dir=_sd_scripts(tmp_path))
    fake_gpu = [
        lambda: Check("torch / CUDA", True, "x"),
        lambda: Check("onnxruntime", True, "x"),
    ]
    checks = gather_checks(
        ws,
        env_path=tmp_path / ".env",
        gpu_probes=fake_gpu,
        settings=settings,
        train_python_probe=lambda _s: Check("trainer torch", True, "arch_list=sm_120"),
    )
    names = [c.name for c in checks]
    assert "sd-scripts" in names and "base model" in names
    assert names[-1] == "trainer torch"  # appended last


def test_format_doctor_report_all_ok() -> None:
    report = format_doctor_report([Check("python", True, "3.12.0"), Check("gpu", True, "ok")])
    assert "[ok]" in report
    assert "All good" in report


def test_format_doctor_report_with_failures() -> None:
    report = format_doctor_report([Check("python", True, "3.12"), Check("gpu", False, "no cuda")])
    assert "[--]" in report
    assert "need attention" in report


def test_format_doctor_report_empty() -> None:
    report = format_doctor_report([])
    assert "Environment check:" in report
    assert "All good" in report  # vacuously
