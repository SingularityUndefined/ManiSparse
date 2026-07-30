"""CLIME precision-matrix estimators.

The package provides two implementations:

* ``clime_cpu``: exact column-wise linear programming through SciPy.
* ``clime_torch``: GPU-capable PyTorch split-ADMM approximation.
"""

from .clime_cpu import ClimeCPUResult, clime_cpu
from .clime_torch import ClimeTorchResult, clime_torch

__all__ = [
    "ClimeCPUResult",
    "ClimeTorchResult",
    "clime_cpu",
    "clime_torch",
]
