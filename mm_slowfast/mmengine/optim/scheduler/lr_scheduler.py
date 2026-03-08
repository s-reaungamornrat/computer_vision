from __future__ import annotations

from .param_scheduler import LinearParamScheduler, CosineAnnealingParamScheduler

class LRSchedulerMixin:
    def __init__(self, optimizer, *args, **kwargs):
        super().__init__(optimizer, 'lr', *args, **kwargs)

class LinearLR(LRSchedulerMixin, LinearParamScheduler):
    """
    Linear LR
    """
class CosineAnnealingLR(LRSchedulerMixin, CosineAnnealingParamScheduler):
    """
    Cosine
    """