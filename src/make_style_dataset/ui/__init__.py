"""Local web UI for the style-dataset pipeline (S8).

Split in two so the quality gate stays Gradio-free:

* :mod:`make_style_dataset.ui.service` — pure, importable, fully-tested helpers
  (no Gradio import): persist uploads, stream stage progress, build galleries,
  zip the dataset, derive run settings.
* :mod:`make_style_dataset.ui.app` — the thin Gradio wiring (coverage-omit, like
  :mod:`make_style_dataset.cli`). Importing it requires the optional ``ui``
  dependency-group; nothing here imports it eagerly.
"""

from __future__ import annotations
