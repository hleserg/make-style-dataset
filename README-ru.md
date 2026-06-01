# make-style-dataset

> Конвейер: страницы комиксов → датасет стиля для LoRA под kohya

[English version](README.md)

Превращает набор страниц комиксов в готовый для
[kohya_ss](https://github.com/bmaltais/kohya_ss) датасет для обучения LoRA на
**стиль**. Инструмент запускает линейный конвейер на файлах: детекция и нарезка
панелей, маски и инпейнт спич-баблов, дедупликация и фильтр по размеру, затем
кэптионинг и раскладка финальной папки датасета.

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

> **Статус:** каркас S0 — внутренности стадий пока заглушки, создающие свои
> выходные папки; алгоритмы появятся в задачах по стадиям.

## Быстрый старт

```bash
uv sync --all-extras                 # создать .venv и поставить всё
cp .env.example .env                 # настроить воркспейс/триггер/пороги (опц.)
uv run make-style-dataset --version
uv run make-style-dataset run-all    # прогнать весь конвейер
make check                           # ворота Definition-of-Done
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
`dedup_hamming_distance`, `min_side_px`, `target_side`) и флаги стадий
(`APP_RUN_*`), которые управляют тем, что выполнит `run-all`.

## Инструменты

uv (окружение/зависимости), ruff (линт+формат), pyright (типы),
pytest (тесты, ≥90%), bandit/pip-audit (безопасность), pre-commit,
commitizen (conventional commits → версия + changelog), Sentry (приватность
по умолчанию).

## Лицензия

MIT — см. [LICENSE](LICENSE).
