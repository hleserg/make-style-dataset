"""Unit tests for the train stage.

Exercise the pure dataset-TOML / argv / progress-parsing logic, the backend
factory, and the full ``run()`` orchestration with a fake trainer — so no torch,
no sd-scripts and no GPU subprocess are needed (the real
:class:`~make_style_dataset.stages.train.SdScriptsTrainer` is ``pragma: no
cover``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from make_style_dataset.config import Settings
from make_style_dataset.pipeline import run_stage
from make_style_dataset.stages import train
from make_style_dataset.stages.base import StageContext
from make_style_dataset.workspace import Workspace


class FakeTrainer:
    """Records the plan and writes a stand-in .safetensors (no subprocess)."""

    def __init__(self) -> None:
        self.plan: train.TrainPlan | None = None

    def train(self, plan: train.TrainPlan) -> Path:
        self.plan = plan
        plan.output_path.parent.mkdir(parents=True, exist_ok=True)
        plan.output_path.write_bytes(b"FAKE_LORA")
        return plan.output_path


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {"workspace": tmp_path / "ws"}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _seed_dataset(ws: Workspace, settings: Settings) -> Path:
    """Create a kohya dataset folder with one png + caption sidecar."""
    dataset = ws.training_dir(settings.dataset_repeats, settings.trigger_token)
    dataset.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), "white").save(dataset / "p1.png")
    (dataset / "p1.txt").write_text(f"{settings.trigger_token}, 1girl\n", encoding="utf-8")
    return dataset


# --- TrainProgress --------------------------------------------------------


def test_progress_fraction() -> None:
    assert train.TrainProgress(step=50, total_steps=200).fraction == 0.25
    assert train.TrainProgress(epoch=1, total_epochs=4).fraction is None
    assert train.TrainProgress(step=5, total_steps=0).fraction is None


# --- parse_progress -------------------------------------------------------


def test_parse_progress_epoch_banner() -> None:
    p = train.parse_progress("epoch 2/10")
    assert p is not None and p.epoch == 2 and p.total_epochs == 10
    assert p.step is None


def test_parse_progress_training_bar_with_loss() -> None:
    line = "steps:  10%|#1        | 200/2000 [01:30<13:30,  2.22it/s, avr_loss=0.0987]"
    p = train.parse_progress(line)
    assert p is not None
    assert (p.step, p.total_steps) == (200, 2000)
    assert p.loss == pytest.approx(0.0987)


def test_parse_progress_loss_scientific_notation() -> None:
    p = train.parse_progress("steps: 1/10 [00:01<00:09, avr_loss=1e-05]")
    assert p is not None and p.loss == pytest.approx(1e-05)


def test_parse_progress_caching_bar_is_ignored() -> None:
    # latent caching prints N/total with no avr_loss and no time bracket
    assert train.parse_progress("steps: 12 (3/45)") is None


def test_parse_progress_avr_loss_without_step_bracket() -> None:
    assert train.parse_progress("note avr_loss=0.5 (no bar here)") is None


def test_parse_progress_avr_loss_present_but_unparseable_value() -> None:
    p = train.parse_progress("steps: 5/10 [00:10<00:20, avr_loss=")
    assert p is not None and (p.step, p.total_steps) == (5, 10) and p.loss is None


def test_parse_progress_unrelated_line() -> None:
    assert train.parse_progress("loading model ...") is None


# --- dataset TOML ---------------------------------------------------------


def test_build_dataset_toml_has_mandatory_keys(tmp_path: Path) -> None:
    toml = train.build_dataset_toml(
        image_dir=tmp_path / "10_style",
        num_repeats=10,
        resolution=512,
        batch_size=1,
        max_bucket_reso=1024,
    )
    assert "caption_extension = '.txt'" in toml  # kohya default is .caption
    assert "keep_tokens = 1" in toml
    assert "shuffle_caption = false" in toml
    assert "num_repeats = 10" in toml
    assert "resolution = 512" in toml
    assert "max_bucket_reso = 1024" in toml
    assert "bucket_reso_steps = 64" in toml
    assert (tmp_path / "10_style").resolve().as_posix() in toml


def test_max_bucket_reso_per_family() -> None:
    assert train._max_bucket_reso("sd15") == 1024
    assert train._max_bucket_reso("sdxl") == 1536
    assert train._max_bucket_reso("flux") == 1536


# --- build_train_args -----------------------------------------------------


def test_build_train_args_sd15_common_and_specific(tmp_path: Path) -> None:
    settings = _settings(tmp_path, train_base_model="/models/sd15.safetensors")
    args = train.build_train_args(
        settings,
        dataset_config=tmp_path / "d.toml",
        output_dir=tmp_path / "out",
        output_name="comicstyle",
    )
    assert "--pretrained_model_name_or_path=/models/sd15.safetensors" in args
    assert "--network_module=networks.lora" in args
    assert "--output_name=comicstyle" in args
    assert "--save_model_as=safetensors" in args
    assert "--sdpa" in args
    assert "--gradient_checkpointing" in args  # default true
    assert "--cache_latents" in args and "--cache_latents_to_disk" in args
    assert "--save_every_n_epochs=1" in args
    assert "--clip_skip=2" in args  # SD 1.5 only
    assert "--optimizer_type=AdamW8bit" in args
    assert "--xformers" not in args  # never (not installed)


def test_build_train_args_conditionals_off(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        train_base_model="/m.safetensors",
        train_gradient_checkpointing=False,
        train_cache_latents=False,
        train_save_every_n_epochs=0,
    )
    args = train.build_train_args(
        settings, dataset_config=tmp_path / "d.toml", output_dir=tmp_path / "o", output_name="s"
    )
    assert "--gradient_checkpointing" not in args
    assert not any(a.startswith("--cache_latents") for a in args)
    assert not any(a.startswith("--save_every_n_epochs") for a in args)
    assert not any(a.startswith("--logging_dir") for a in args)


def test_build_train_args_logging_dir(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path, train_base_model="/m.safetensors", train_logging_dir=tmp_path / "logs"
    )
    args = train.build_train_args(
        settings, dataset_config=tmp_path / "d.toml", output_dir=tmp_path / "o", output_name="s"
    )
    assert any(a.startswith("--logging_dir=") for a in args)
    assert "--log_with=tensorboard" in args


def test_build_train_args_empty_base_model_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train_base_model"):
        train.build_train_args(
            _settings(tmp_path),
            dataset_config=tmp_path / "d.toml",
            output_dir=tmp_path / "o",
            output_name="s",
        )


def test_build_train_args_unknown_family_raises(tmp_path: Path) -> None:
    # sd3 is a real sd-scripts family we have not wired -> unsupported
    settings = _settings(tmp_path, train_model_type="sd3", train_base_model="/m.safetensors")
    with pytest.raises(NotImplementedError, match="sd3"):
        train.build_train_args(
            settings, dataset_config=tmp_path / "d.toml", output_dir=tmp_path / "o", output_name="s"
        )


def _flux_args(tmp_path: Path, **overrides: object) -> list[str]:
    base: dict[str, object] = {
        "train_model_type": "flux",
        "train_base_model": "/flux/dit.safetensors",
        "train_flux_clip_l": "/flux/clip_l.safetensors",
        "train_flux_t5xxl": "/flux/t5xxl.safetensors",
        "train_flux_ae": "/flux/ae.safetensors",
    }
    base.update(overrides)
    settings = _settings(tmp_path, **base)
    return train.build_train_args(
        settings, dataset_config=tmp_path / "d.toml", output_dir=tmp_path / "o", output_name="s"
    )


def test_build_train_args_flux_recipe(tmp_path: Path) -> None:
    args = _flux_args(tmp_path)
    assert "--clip_l=/flux/clip_l.safetensors" in args
    assert "--t5xxl=/flux/t5xxl.safetensors" in args
    assert "--ae=/flux/ae.safetensors" in args
    assert "--network_train_unet_only" in args
    assert "--guidance_scale=1.0" in args
    assert "--timestep_sampling=flux_shift" in args
    assert "--model_prediction_type=raw" in args
    assert "--fp8_base" in args
    assert "--cache_text_encoder_outputs" in args  # default cache_latents True
    assert "--blocks_to_swap=18" in args  # default
    assert "--no_half_vae" not in args  # SDXL-only
    assert not any(a.startswith("--clip_skip") for a in args)  # SD 1.5-only


def test_build_train_args_flux_missing_components_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="APP_TRAIN_FLUX_T5XXL"):
        _flux_args(tmp_path, train_flux_t5xxl="")


def test_build_train_args_flux_no_cache_no_swap(tmp_path: Path) -> None:
    args = _flux_args(tmp_path, train_cache_latents=False, train_flux_blocks_to_swap=0)
    assert not any(a.startswith("--cache_text_encoder_outputs") for a in args)
    assert not any(a.startswith("--blocks_to_swap") for a in args)
    # the constant recipe flags are still present
    assert "--fp8_base" in args and "--timestep_sampling=flux_shift" in args


def _sdxl_args(tmp_path: Path, **overrides: object) -> list[str]:
    settings = _settings(
        tmp_path, train_model_type="sdxl", train_base_model="/m.safetensors", **overrides
    )
    return train.build_train_args(
        settings, dataset_config=tmp_path / "d.toml", output_dir=tmp_path / "o", output_name="s"
    )


def test_build_train_args_sdxl_unet_only_default(tmp_path: Path) -> None:
    args = _sdxl_args(tmp_path)
    assert "--no_half_vae" in args  # mandatory for SDXL
    assert "--network_module=networks.lora" in args
    assert "--network_train_unet_only" in args  # default train_unet_only=True
    assert "--cache_text_encoder_outputs" in args
    assert "--cache_text_encoder_outputs_to_disk" in args
    assert not any(a.startswith("--clip_skip") for a in args)  # SD 1.5 only
    assert not any(a.startswith("--text_encoder_lr") for a in args)


def test_build_train_args_sdxl_train_text_encoders(tmp_path: Path) -> None:
    args = _sdxl_args(tmp_path, train_unet_only=False, train_text_encoder_lr=5e-5)
    assert "--no_half_vae" in args
    assert "--network_train_unet_only" not in args
    # TE-output caching is forbidden with TE training (kohya asserts) -> dropped
    assert not any(a.startswith("--cache_text_encoder_outputs") for a in args)
    # nargs=* : the flag is followed by two separate values
    idx = args.index("--text_encoder_lr")
    assert args[idx + 1] == "5e-05" and args[idx + 2] == "5e-05"


def test_build_train_args_sdxl_train_te_without_lr(tmp_path: Path) -> None:
    # train_unet_only False but no TE lr set -> no --text_encoder_lr, no cache flags
    args = _sdxl_args(tmp_path, train_unet_only=False)
    assert "--network_train_unet_only" not in args
    assert not any(a.startswith("--text_encoder_lr") for a in args)
    assert not any(a.startswith("--cache_text_encoder_outputs") for a in args)


# --- launch command / interpreter / output name ---------------------------


def test_resolve_python_default_is_venv(tmp_path: Path) -> None:
    settings = _settings(tmp_path, train_sd_scripts_dir=Path("/home/serg/sd-scripts"))
    assert train.resolve_python(settings) == "/home/serg/sd-scripts/venv/bin/python"


def test_resolve_python_explicit_override(tmp_path: Path) -> None:
    settings = _settings(tmp_path, train_python="/opt/py/bin/python")
    assert train.resolve_python(settings) == "/opt/py/bin/python"


def test_build_launch_command_shape(tmp_path: Path) -> None:
    settings = _settings(tmp_path, train_mixed_precision="bf16")
    cmd = train.build_launch_command(settings, ["--foo=1"])
    assert cmd[1:3] == ["-m", "accelerate.commands.launch"]
    assert "--num_processes" in cmd and "--dynamo_backend" in cmd
    assert "train_network.py" in cmd  # sd15 entrypoint
    assert cmd[-1] == "--foo=1"  # script args appended last
    # mixed precision is handed to accelerate too
    assert cmd[cmd.index("--mixed_precision") + 1] == "bf16"


def test_build_launch_command_unknown_family_raises(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        train.build_launch_command(_settings(tmp_path, train_model_type="bogus"), [])


def test_build_launch_command_sdxl_entrypoint(tmp_path: Path) -> None:
    cmd = train.build_launch_command(_settings(tmp_path, train_model_type="sdxl"), ["--foo=1"])
    assert "sdxl_train_network.py" in cmd
    assert "train_network.py" not in cmd  # exact element, not the sd15 entrypoint


def test_build_launch_command_flux_entrypoint(tmp_path: Path) -> None:
    cmd = train.build_launch_command(_settings(tmp_path, train_model_type="flux"), [])
    assert "flux_train_network.py" in cmd


def test_resolve_output_name(tmp_path: Path) -> None:
    assert train.resolve_output_name(_settings(tmp_path, trigger_token="foo")) == "foo"
    assert (
        train.resolve_output_name(_settings(tmp_path, train_output_name=" bar "))  # stripped
        == "bar"
    )


# --- factory --------------------------------------------------------------


def test_make_trainer_returns_sdscripts(tmp_path: Path) -> None:
    assert isinstance(train.make_trainer(_settings(tmp_path)), train.SdScriptsTrainer)


# --- run() orchestration --------------------------------------------------


def test_run_trains_into_lora_dir(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(
        tmp_path, train_base_model="/m.safetensors", trigger_token="mystyle", dataset_repeats=12
    )
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    _seed_dataset(ws, settings)
    fake = FakeTrainer()
    monkeypatch.setattr(train, "make_trainer", lambda *_a, **_k: fake)

    result = train.run(StageContext(workspace=ws, settings=settings))

    assert result.produced == 1
    assert result.output_dir == ws.lora
    assert (ws.lora / "mystyle.safetensors").is_file()
    assert (ws.lora / "dataset.toml").is_file()
    # the plan the trainer received points at the right entrypoint + dataset
    assert fake.plan is not None
    assert "train_network.py" in fake.plan.command
    assert fake.plan.cwd == settings.train_sd_scripts_dir
    assert fake.plan.output_path == ws.lora / "mystyle.safetensors"


def test_run_sdxl_uses_sdxl_entrypoint(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(
        tmp_path, train_model_type="sdxl", train_base_model="/m.safetensors", train_resolution=1024
    )
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    _seed_dataset(ws, settings)
    fake = FakeTrainer()
    monkeypatch.setattr(train, "make_trainer", lambda *_a, **_k: fake)

    result = train.run(StageContext(workspace=ws, settings=settings))

    assert result.produced == 1
    assert fake.plan is not None
    assert "sdxl_train_network.py" in fake.plan.command
    assert "--no_half_vae" in fake.plan.command
    # SDXL/Flux use the larger bucket ceiling in the generated TOML
    assert "max_bucket_reso = 1536" in (ws.lora / "dataset.toml").read_text(encoding="utf-8")


def test_run_flux_uses_flux_entrypoint(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(
        tmp_path,
        train_model_type="flux",
        train_base_model="/flux/dit.safetensors",
        train_flux_clip_l="/flux/clip_l.safetensors",
        train_flux_t5xxl="/flux/t5xxl.safetensors",
        train_flux_ae="/flux/ae.safetensors",
        train_resolution=1024,
    )
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    _seed_dataset(ws, settings)
    fake = FakeTrainer()
    monkeypatch.setattr(train, "make_trainer", lambda *_a, **_k: fake)

    result = train.run(StageContext(workspace=ws, settings=settings))

    assert result.produced == 1
    assert fake.plan is not None
    assert "flux_train_network.py" in fake.plan.command
    assert "--fp8_base" in fake.plan.command
    assert "--clip_l=/flux/clip_l.safetensors" in fake.plan.command


def test_run_without_dataset_raises(tmp_path: Path) -> None:
    settings = _settings(tmp_path, train_base_model="/m.safetensors")
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    with pytest.raises(ValueError, match="no dataset images"):
        train.run(StageContext(workspace=ws, settings=settings))


def test_run_is_idempotent_via_marker(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, train_base_model="/m.safetensors", trigger_token="s")
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    _seed_dataset(ws, settings)
    monkeypatch.setattr(train, "make_trainer", lambda *_a, **_k: FakeTrainer())
    ctx = StageContext(workspace=ws, settings=settings)

    first = run_stage(train.STAGE, ctx)
    second = run_stage(train.STAGE, ctx)
    assert first.skipped is False
    assert second.skipped is True
    assert "force" in second.reason


def test_stage_metadata() -> None:
    assert train.STAGE.name == train.NAME == "train"
    assert train.STAGE.flag == "run_train"
    assert train.STAGE.component.startswith("stage:")
