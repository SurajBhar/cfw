"""
Evaluation module for CFW.

This module provides comprehensive evaluation utilities including metrics computation,
model evaluation, and testing utilities.
"""


from .metrics import (
    calculate_balanced_accuracy,
    calculate_accuracy,
    calculate_per_class_recall,
    calculate_confusion_matrix,
    calculate_precision_recall_f1,
    calculate_top_k_accuracy,
    format_metrics
)

from .evaluator import Evaluator

from .testing import (
    test_model,
    test_multiple_checkpoints,
    test_experiment_checkpoints,
    save_test_results,
    print_test_summary,
    find_best_checkpoint,
    compare_experiments,
    print_experiment_comparison
)

from .feature_analysis import (
    compute_metrics_only_analysis,
    compute_feature_metrics,
    validate_feature_quality,
    analyze_feature_file,
    print_feature_metrics,
    compare_feature_sets
)


__all__ = [
    # Metrics
    'calculate_balanced_accuracy',
    'calculate_accuracy',
    'calculate_per_class_recall',
    'calculate_confusion_matrix',
    'calculate_precision_recall_f1',
    'calculate_top_k_accuracy',
    'format_metrics',
    # Evaluator
    'Evaluator',
    # Testing
    'test_model',
    'test_multiple_checkpoints',
    'test_experiment_checkpoints',
    'save_test_results',
    'print_test_summary',
    'find_best_checkpoint',
    'compare_experiments',
    'print_experiment_comparison',
    # Feature Analysis
    'compute_metrics_only_analysis',
    'compute_feature_metrics',
    'validate_feature_quality',
    'analyze_feature_file',
    'print_feature_metrics',
    'compare_feature_sets',
]
