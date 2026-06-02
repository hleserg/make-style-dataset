# Developing on WSL + CUDA

This project is **GPU-bound in its later stages**, so the recommended development
environment is **WSL2 (Ubuntu) with an NVIDIA CUDA driver**, where models can be
cached locally and a generation UI (ComfyUI / AUTOMATIC1111) is available to
validate the trained LoRA. Linux also removes Windows-specific friction the
project has hit (non-ASCII paths breaking `cv2.imread`, missing `jq`/`make`,
non-portable shell hooks).

## Which stages need the GPU

| Stage | GPU | Why |
|-------|-----|-----|
| `panels` (S1) | no | OpenCV contour detection on CPU |
| `bubbles` (S2) | likely | a learned bubble detector/segmenter, if used |
| `inpaint` (S3) | **yes** | diffusion / LaMa inpainting |
| `clean` (S4) | no | perceptual-hash dedup |
| `caption` (S5) | **yes** | vision-language model / tagger (BLIP, WD14) |
| downstream | **yes** | kohya LoRA training + validating output in a generation UI |

Stages S1 and S4 run fine CPU-only; S2/S3/S5 and the end goal want CUDA.

## One-time setup

```bash
# In WSL (Ubuntu), with the NVIDIA driver installed on Windows and `nvidia-smi` working:
git clone https://github.com/hleserg/make-style-dataset.git
cd make-style-dataset

curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv if absent
uv sync --all-extras                              # create .venv, install deps
uv run pre-commit install                         # repo hooks
make check                                         # DoD gate — must be green
```

`make` and `jq` exist on Linux, so the `Makefile` targets and any shell hooks
work directly (unlike the Windows host).

### GPU dependencies (added per stage)

The current dependency set is CPU-only (`opencv-python-headless`, `numpy`,
`Pillow`). The model stages will add CUDA libraries (e.g. `torch`,
`diffusers`/`transformers`, a tagger). Install the CUDA build of torch that
matches the box's driver, e.g.:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
nvidia-smi   # confirm the GPU is visible from WSL
```

Pin exact versions in `pyproject.toml` + the lockfile when a stage lands, and
keep heavy/GPU deps grouped so the CPU-only stages remain installable without them.

## Exposing the local UI to the LAN (WSL networking)

> **THE RULE: a WSL service must listen on `0.0.0.0` inside WSL, never `127.0.0.1`.**

The Gradio UI (`make-style-dataset ui`, default port 7860) must bind `0.0.0.0`, so
launch it as `APP_UI_HOST=0.0.0.0 make-style-dataset ui` (the default `ui_host` is
`127.0.0.1`). Windows reaches a WSL service through the WSL distro's **eth0 IP**
(e.g. `172.22.x.x`) via a `netsh portproxy`; a service bound to WSL **loopback**
(`127.0.0.1`) is invisible to that proxy and unreachable from the host or LAN.

**Reaching it from the Windows host / another LAN machine (Windows 10):**

> WSL **mirrored networking** (`networkingMode=mirrored`, no portproxy needed)
> requires **Windows 11 22H2+**; on Windows 10 the `.wslconfig` line is ignored and
> WSL falls back to NAT — so you need a host-side portproxy + firewall rule.

```powershell
# Elevated PowerShell. Re-derive the WSL IP (it changes when WSL restarts):
$ip = (wsl hostname -I).Trim().Split(' ')[0]; "WSL IP = $ip"
$port = 7860
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$port 2>$null
netsh interface portproxy add    v4tov4 listenaddress=0.0.0.0 listenport=$port connectaddress=$ip connectport=$port
netsh advfirewall firewall delete rule name=WSL_LAN_$port 2>$null
netsh advfirewall firewall add    rule name=WSL_LAN_$port dir=in action=allow protocol=TCP localport=$port remoteip=LocalSubnet
```

Then open `http://<windows-LAN-IP>:<port>` from any LAN machine (or `localhost:<port>`
on the host). To survive reboots/WSL restarts, drive that script from a Scheduled Task
on logon + a short interval; `schtasks /Create` is more robust than
`Register-ScheduledTask` on localized / non-domain Windows. Prefer a LAN-scoped
(`remoteip=LocalSubnet`) firewall rule over disabling the firewall.

## Keep the code OS-agnostic

The pipeline must keep running on Windows too. Two rules learned on the Windows host:

- **Never call `cv2.imread`/`cv2.imwrite` with a path** — it fails on non-ASCII
  paths. Decode via `np.fromfile(path, np.uint8)` + `cv2.imdecode(...)` (see
  `ContourPanelDetector` in `src/make_style_dataset/stages/panels.py`). Harmless
  on Linux, required on Windows.
- Prefer pure stdlib path handling; don't hardcode shell tools in committed code.

## Continuing the pipeline (agent onboarding)

Read these first, in order: `AGENTS.md` (canonical rules), then
`docs/architecture/SYSTEM.md` and `docs/architecture/WORKSPACE.md`.

Definition of Done is a hard gate: `make check` green (ruff + ruff format +
pyright + bandit + pip-audit + pytest with ≥90 % coverage). Conventional
Commits, small reviewable PRs, English commits/comments. On taking a Linear
issue set it **In Progress**; on finishing set **Done** and link the PR.

S1 (`panels`) is the reference implementation for the remaining stages — reuse
its patterns:

- An **injectable detector behind a `Protocol`** so the heavy/optional backend
  is swappable and tests inject a fake.
- **Pure policy/geometry functions over plain data** (filter, classify, route),
  with the heavy dependency (OpenCV, and later torch) **imported lazily** inside
  the adapter — keeps coverage high without the heavy dep on the unit path. See
  the `pure-core-lazy-backend` PLAYBOOK marker.
- **Deterministic, traceable output names** (`<source-stem>_<idx>.png`) so
  re-runs overwrite instead of duplicating and each artifact traces to its source.
- A **`manual_review/` fallback** for inputs the automation should not force.

Next stage is **S2 `bubbles`** (then S3 `inpaint`, S5 `caption`). Pick up the
corresponding Linear issue.

### Set up local hooks in WSL

Don't reuse the Windows hook commands. In WSL run `/update-config` to add
machine-native hooks: format-on-save (`ruff format` on edited `*.py`) and a
pre-PR `make check` gate. These go in `.claude/settings.local.json`
(git-ignored); the shared command allowlist lives in the committed
`.claude/settings.json`.
