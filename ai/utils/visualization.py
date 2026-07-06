"""
Plant Guard AI - Visualization Utilities
==========================================
Helper functions for plotting training curves, prediction results,
GradCAM overlays, and dataset sample grids.
"""

# TODO: Import matplotlib, seaborn, numpy, PIL as needed


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        train_losses (list): Training loss per epoch.
        val_losses (list): Validation loss per epoch.
        train_accs (list): Training accuracy per epoch.
        val_accs (list): Validation accuracy per epoch.
        save_path (str, optional): Path to save the figure.
    """
    # TODO: Implement training curve plotting with matplotlib
    raise NotImplementedError("plot_training_curves() is not yet implemented.")


def plot_sample_grid(images, labels, class_names, n_cols=4, save_path=None):
    """
    Display a grid of sample images with their labels.

    Args:
        images (list): List of image tensors or arrays.
        labels (list): Corresponding label indices.
        class_names (list): Class name strings.
        n_cols (int): Number of columns in the grid.
        save_path (str, optional): Path to save the figure.
    """
    # TODO: Implement image grid visualization
    raise NotImplementedError("plot_sample_grid() is not yet implemented.")


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot a styled confusion matrix heatmap.

    Args:
        cm (numpy.ndarray): Confusion matrix.
        class_names (list): Class label names.
        save_path (str, optional): Path to save the figure.
    """
    # TODO: Implement confusion matrix heatmap with seaborn
    raise NotImplementedError("plot_confusion_matrix() is not yet implemented.")
