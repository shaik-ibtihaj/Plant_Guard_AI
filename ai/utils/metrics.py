"""
Plant Guard AI - Evaluation Metrics
=====================================
Utility functions for computing model evaluation metrics including
accuracy, precision, recall, F1-score, and confusion matrix.
"""

# TODO: Import sklearn.metrics and numpy


def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Args:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.

    Returns:
        float: Accuracy score.
    """
    # TODO: Implement accuracy computation
    raise NotImplementedError("compute_accuracy() is not yet implemented.")


def compute_f1_score(y_true, y_pred, average="weighted"):
    """
    Compute F1 score.

    Args:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
        average (str): Averaging strategy ('weighted', 'macro', 'micro').

    Returns:
        float: F1 score.
    """
    # TODO: Implement F1 score computation
    raise NotImplementedError("compute_f1_score() is not yet implemented.")


def compute_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Compute and return confusion matrix.

    Args:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
        class_names (list, optional): Class label names.

    Returns:
        numpy.ndarray: Confusion matrix.
    """
    # TODO: Implement confusion matrix computation
    raise NotImplementedError("compute_confusion_matrix() is not yet implemented.")
