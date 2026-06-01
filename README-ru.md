# make-style-dataset

> Конвейер: страницы комиксов → датасет стиля для LoRA под kohya

[English version](README.md)

Превращает набор страниц комиксов в готовый для
[kohya_ss](https://github.com/bmaltais/kohya_ss) датасет для обучения LoRA на
**стиль**. Инструмент запускает линейный конвейер на файлах: детекция и нарезка
панелей, маски и инпейнт спич-баблов, дедупликация и фильтр по размеру, затем
кэптионинг и раскладка финальной папки датасета.

> **Не программист?** Начните с понятной
> [**пошаговой инструкции**](docs/USER_GUIDE-ru.md) ([EN](docs/USER_GUIDE.md)):
> одна команда установки (`bash scripts/setup.sh`), положить страницы, запустить.

---

## Конвейер

Шесть стадий, каждая читает одну папку воркспейса и пишет в следующую. `00_pages`
— это вход; остальные пять — подкоманды CLI:

| # | Стадия | Читает → пишет | Что делает |
|---|--------|----------------|------------|
| 0 | *(pages)* | → `00_pages/` | Исходные страницы комикса, которые вы кладёте сами. |
| 1 | `panels` | `00_pages` → `01_panels` | Детектит панели и нарезает каждую страницу. |
| 2 | `bubbles` | `01_panels` → `02_masks` | Находит спич-баблы, пишет маски удаления. |
| 3 | `inpaint` | `01_panels`+`02_masks` → `03_inpainted` | Закрашивает баблы (инпейнт). |
| 4 | `clean` | `03_inpainted` → `04_clean` | Выкидывает почти-дубли и слишком мелкие панели. |
| 5 | `caption` | `04_clean` → `05_dataset/<N>_<trigger>/` | Кэптионит и раскладывает датасет под kohya. |

Раннер **идемпотентен**: каждая стадия кладёт маркер `.stage_complete` и
пропускается при повторном запуске, если не передан `--force`. См.
[контракт раскладки воркспейса](docs/architecture/WORKSPACE.md) и
[обзор системы](docs/architecture/SYSTEM.md).

Все шесть стадий реализованы. `panels` и `clean` работают на CPU; модельные
стадии — `bubbles` (YOLOv8-seg + EasyOCR), `inpaint` (ONNX Big-LaMa) и
`caption` (WD14 ViT v3, ONNX) — требуют опциональной группы зависимостей
**`gpu`** (см. [GPU-стадии](#gpu-стадии)). `run-all` печатает сводку: сколько
артефактов произвела каждая стадия.

## Быстрый старт

**Простой путь** (ставит всё, создаёт воркспейс, проверяет GPU):

```bash
bash scripts/setup.sh                # установка одной командой; --no-gpu чтобы без GPU-стека
#   затем положите страницы в workspace/00_pages/
uv run make-style-dataset run-all    # собрать датасет
```

**Ручной путь** (для разработчиков):

```bash
uv sync --all-extras                 # создать .venv + dev-инструменты (CPU-стадии готовы)
uv run make-style-dataset init       # создать воркспейс + засеять .env
uv run make-style-dataset doctor     # проверить Python / GPU / воркспейс
make check                           # ворота Definition-of-Done

# Для модельных стадий (bubbles/inpaint/caption) добавьте GPU-зависимости:
uv sync --all-extras --group gpu     # torch (cu128) + onnxruntime + ultralytics/easyocr (неск. ГБ)
uv run make-style-dataset run-all    # прогнать весь конвейер
```

## Использование

```bash
# Запустить одну стадию или весь конвейер:
uv run make-style-dataset panels
uv run make-style-dataset run-all
uv run make-style-dataset run-all --help     # покажет все стадии

# Полезные флаги:
uv run make-style-dataset run-all --workspace /data/comics   # сменить корень воркспейса
uv run make-style-dataset clean --force                      # перезапустить готовую стадию
```

Конфигурация — через переменные окружения (префикс `APP_`, см. `.env.example`):
корень воркспейса, триггер-токен, kohya-повторы, пороги (`min_panel_area`,
`dedup_hamming_distance`, `min_side_px`, `target_side`), выбор бэкендов
(`inpaint_backend`, `caption_backend`) и флаги стадий (`APP_RUN_*`), которые
управляют тем, что выполнит `run-all`.

## GPU-стадии

Модельные стадии скачивают веса с Hugging Face (закреплены по коммиту) при
первом запуске и используют GPU, если он доступен:

| Стадия | Модель | Бэкенд |
|--------|--------|--------|
| `bubbles` | `kitsumed/yolov8m_seg-speech-bubble` + EasyOCR | ultralytics (torch, cu128) |
| `inpaint` | `Carve/LaMa-ONNX` (Big-LaMa) | onnxruntime |
| `caption` | `SmilingWolf/wd-vit-tagger-v3` | onnxruntime |

Ставятся через `uv sync --all-extras --group gpu`. Они живут в группе
зависимостей PEP 735 (а не в extra), поэтому обычный `uv sync --all-extras` (CI
и CPU-стадии) остаётся лёгким. CUDA-сборка torch (`cu128`, под Blackwell/RTX
50xx) закреплена через `[tool.uv.sources]`; на macOS lock падает на CPU-сборку.
onnxruntime откатывается на CPU, если не находит свои CUDA-библиотеки (голые pip
-колёса не везут cuDNN/`libcublasLt` — их даёт CUDA-базовый образ или локальная
установка CUDA/cuDNN).

**Распределение по железу:** `panels`/`clean` (CPU) — где угодно;
`bubbles`/`inpaint`/`caption` — на GPU-хосте. Контейнеризация тяжёлых стадий под
GPU-машину — запланированный следующий шаг.

## Инструменты

uv (окружение/зависимости), ruff (линт+формат), pyright (типы),
pytest (тесты, ≥90%), bandit/pip-audit (безопасность), pre-commit,
commitizen (conventional commits → версия + changelog), Sentry (приватность
по умолчанию).

## Лицензия

MIT — см. [LICENSE](LICENSE).
