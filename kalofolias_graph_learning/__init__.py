"""Graph learning from smooth signals after Kalofolias (AISTATS 2016)."""

from .kalofolias import KalofoliasResult, learn_graph_from_smooth_signals
from .torch_module import KalofoliasGraphLearningModule

__all__ = [
    "KalofoliasResult",
    "KalofoliasGraphLearningModule",
    "learn_graph_from_smooth_signals",
]
