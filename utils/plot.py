import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_accuracy_curve_by_exp_group(fname: str, title: str = '', acc_line: list[float] = [.8, .82], **kwargs):
    """
    Plot accuracy curve by experiment group.

    Args:
        fname (str): file name of the plot
        title (str, optional): title of the plot. Defaults to ''.
        acc_line (list[float], optional): result baseline. Defaults to [.8, .82].
    """
    fig = plt.figure(figsize=(8, 4.5))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    for label, data in kwargs.items():
        plt.plot(
            range(1, len(data)+1), data,
            '--' if 'test' in label else '-',
            label=label
        )

    plt.legend(
        loc='best', bbox_to_anchor=(1.0, 1.0, 0.2, 0),
        fancybox=True, shadow=True
    )

    if acc_line:
        plt.hlines(acc_line, 1, len(data)+1, linestyles='dashed', colors=(0, 0, 0, 0.8))

    plt.savefig(fname, dpi=300, bbox_inches="tight")


def plot_confusion_matrix(fname: str, title: str = '', cm: np.ndarray = None, classes: int = None):
    """
    Plot confusion matrix.

    Args:
        fname (str): file name of the plot
        title (str, optional): title of the plot. Defaults to ''.
        cm (np.ndarray, optional): confusion matrix. Defaults to None.
        classes (int, optional): num of classes. Defaults to None.
    """
    disp = ConfusionMatrixDisplay(cm, display_labels=np.arange(classes))
    disp.plot(cmap='Blues', include_values=True)
    disp.ax_.set_title(title)
    plt.savefig(fname, dpi=300, bbox_inches="tight")
