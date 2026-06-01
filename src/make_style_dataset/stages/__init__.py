"""Pipeline stages: one module per transform between workspace folders."""

from make_style_dataset.stages.base import Stage, StageContext, StageResult

__all__ = ["Stage", "StageContext", "StageResult"]
