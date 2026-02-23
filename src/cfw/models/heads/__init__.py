"""
Classification heads package for CFW.

This package contains classification head implementations:
- Linear: Simple linear classifier for frozen features
- MLP: Multi-layer perceptron classifier with additional capacity
"""


from .linear import (
    LinearClassifier,
    LinearClassifierWithBackbone,
)
from .mlp import (
    MLPClassifier,
    MLPClassifierWithBackbone,
)

__all__ = [
    'LinearClassifier',
    'LinearClassifierWithBackbone',
    'MLPClassifier',
    'MLPClassifierWithBackbone',
]
