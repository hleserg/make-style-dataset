# HLE-803 — статус и что дальше

Живая рабочая записка по задаче **HLE-803** (стилевая LoRA `cmcstyle`). Родитель —
HLE-802. Обновлено 2026-06-02.

> **СТАТУС:** датасет готов, механизм проверен репетицией, **ночной запуск обучения
> запланирован на среду 2026-06-03 02:00 (один раз).** Утром — снять диагностику + eval.

---

## Часть 1 — для человека (простым языком)

### О чём задача
Обучить **стилевую LoRA `cmcstyle`** — адаптер к Flux.1-dev, накладывающий манеру
комикса (пейнтерли-историческая BD) на **любой** сюжет, **не** протаскивая конкретных
персонажей/сцен/панелей.

### Кто что делает
- **Другой агент** — датасет (нарезка, бабблы, чистка) + **капшнинг** (новая VLM-проза
  через Gemini-прокси). Его зона, не лезу.
- **Я (style-lora агент)** — обучение + eval: настройки, рецепт, eval-инструмент,
  лаунчер; запускаю обучение.

### Что готово
1. **Тулинг:** **ai-toolkit** (не kohya — он падает по памяти на Flux на нашей 16 ГБ).
2. **Рецепт + настройки + протокол eval:** `docs/features/style-lora-cmcstyle.md` (+ `-ru.md`).
3. **Eval-инструмент:** `scripts/eval_style_lora.py` (сетка «сила LoRA × промт»).
4. **Лаунчер ночного запуска:** `scripts/train_cmcstyle_aitk.sh` (рабочая копия, которую
   гонит таймер: `/home/serg/cmcstyle_night/launch.sh`).
5. **Датасет готов:** `…/workspace/05_dataset/10_cmcstyle` — 116 картинок, VLM-капшны
   прозой с триггером `cmcstyle`, без стиль-слов. ✓
6. **Репетиция (steps=2) прошла:** вся цепочка работает под systemd (CUDA, бакеты,
   flip-аугментация, сейв) → вышел **не пустой LoRA 344 МБ / 988 тензоров** за 366 с;
   ~9.7 с/it → полный прогон ≈ **5–5.5 ч**.

### Ночной запуск (запланирован)
- **Таймер `cmcstyle-night.timer` → среда 02:00, один раз** (не повторяется),
  запускает `scripts/train_cmcstyle_aitk.sh night` (steps=2000, чекпоинты каждые 250).
- В 02:00 лаунчер **сам проверит**: датасет готов+стабилен, GPU свободна (<2 ГБ), CUDA.
  Если что-то не так — **чисто прервётся и запишет причину** (не запустит обучение зря).

### Что нужно ОТ ТЕБЯ
1. **Не выключай и не ребутай WSL сегодня ночью** — таймер (transient) не переживёт
   перезагрузку.
2. Если ночью будет работать сосед (его char-LoRA на GPU) — мой запуск это увидит и
   прервётся; перепланируем.
3. **Утром пингни меня** — сниму диагностику и прогоню eval.

### Что я сделаю утром
1. Сниму диагностику: пик/запас VRAM (`gpu.csv`), s/it, кривая loss, время загрузки,
   размер+тензоры LoRA (`meta.json`, `train.log`).
2. Прогоню `scripts/eval_style_lora.py` по чекпоинтам в ComfyUI → сетка «вес × промт» →
   **выберу лучший чекпоинт и силу LoRA**, проверю утечку лиц/рамок/бабблов.
3. Отдам `cmcstyle.safetensors` + сетку-отчёт + зафиксированные настройки + рекомендации.

### Команды
```bash
# отменить ночной запуск:
systemctl stop cmcstyle-night.timer
# утром — где результаты (символ. ссылка на последний прогон):
ls -l /home/serg/cmcstyle_night/latest/        # meta.json, train.log, gpu.csv, out/cmcstyle_flux/
# eval (ComfyUI запущен):
python scripts/eval_style_lora.py --lora <чекпоинт>.safetensors
```

### Где что лежит
| Что | Путь |
|---|---|
| Рецепт + настройки + eval-протокол | `docs/features/style-lora-cmcstyle.md` (+ `-ru.md`) |
| Eval-инструмент | `scripts/eval_style_lora.py` |
| Лаунчер (репо-копия / рабочая) | `scripts/train_cmcstyle_aitk.sh` / `/home/serg/cmcstyle_night/launch.sh` |
| Диагностика прогонов | `/home/serg/cmcstyle_night/runs/<ts>_night/` (+ `latest`) |
| Датасет | `…/make-style-dataset/workspace/05_dataset/10_cmcstyle` |
| Хендоффы соседу | `make-char-dataset/FLUX_TRAINING_HANDOFF.md`, `CAPTIONING_SYSTEM.md` |

---

## Часть 2 — для агента (резюме на возобновление)

**Роль:** style-lora агент (HLE-803), обучающая половина. Датасет+капшнинг — другой
агент, **не трогать** (`workspace/`, стадии pipeline, `caption.py`). Память:
`hle803-style-lora-agent-scope`, `train-stage-sd-scripts`, `gemini-proxy-and-vlm-caption`.

**В `main`:** PR #22 рецепт-доки, #23 eval-харнесс, #24 этот статус, + лаунчер
`scripts/train_cmcstyle_aitk.sh`. Worktree `.claude/worktrees/hle-803-style-lora`.

**Тулинг = ai-toolkit (НЕ kohya).** `/home/serg/ai-toolkit` v0.9.14, venv torch
2.12.dev+cu128 sm_120. Запуск **онлайн** (НЕ `HF_HUB_OFFLINE=1`). Грабли: `qtype:
qfloat8 + qtype_te: qfloat8 + low_vram: true` (qint4 ломается); `dtype: bf16`;
`disable_sampling: true` (in-training sampling → OOM, eval пост-хок в ComfyUI);
nvidia-smi на WSL — в `/usr/lib/wsl/lib` (нужен в PATH для systemd). ~9.7 с/it @512 rank32.

**Ночной запуск (2026-06-03 02:00, one-shot).** `cmcstyle-night.timer` →
`scripts/train_cmcstyle_aitk.sh night`. Лаунчер сам проверяет предусловия и абортит+логирует.
- **Снять результаты в один шаг:** `/home/serg/cmcstyle_night/latest/` → `meta.json`
  (status/тайминги/safetensors bytes+tensors), `train.log`, `gpu.csv`, `env.txt`,
  `out/cmcstyle_flux/` (чекпоинты). Затем eval пост-хок: `scripts/eval_style_lora.py --lora …`.
- **Отмена:** `systemctl stop cmcstyle-night.timer`. Caveats: transient-таймер не
  переживёт ребут WSL; репетиция оставила безвредный переиспользуемый `_latent_cache/`
  в датасет-папке соседа.

**После успеха + eval:** закрыть HLE-803 как Done. **Открытое:** триггер по умолчанию в
репо `comicstyle`, но датасет уже на `cmcstyle` (см. `10_cmcstyle`). После сквош-мержа
ветку синхронизировать `git reset --hard origin/main`.
