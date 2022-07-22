import matplotlib.pyplot as plt


def plot_accuracy_curve_by_exp_group(fname, title='', acc_line=[.85, .87], **kwargs):
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
