import numpy as np

from typing import List, Dict, Tuple
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


def transform_label_to_binary(
    true_vs_predicted: List[Dict[str, np.ndarray]]
) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:

    unique_values = np.unique(true_vs_predicted[0]["true"])
    if len(unique_values) > 2:
        raise Exception("This is not binary classification")
    if len(unique_values) != 2:
        mapped_to_0 = unique_values[0]
        print(f"Label {mapped_to_0} in mapped to 0, another label is mapped to 1")
    else:
        mapped_to_0 = unique_values[0]
        mapped_to_1 = unique_values[1]
        print(f"Label {mapped_to_0} in mapped to 0, label {mapped_to_1} in mapped to 1")
    for record in true_vs_predicted:
        for index in range(len(record["true"])):
            record["true"][index] = 0 if record["true"][index] == mapped_to_0 else 1
            record["predicted"][index] = (
                0 if record["predicted"][index] == mapped_to_0 else 1
            )

    return (true_vs_predicted, unique_values)


def roc_and_auc_metrics(true_vs_predicted: List[Dict[str, np.ndarray]]) -> None:

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(1)

    for index, record in enumerate(true_vs_predicted):
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(record["true"], record["predicted"])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(
            fpr,
            tpr,
            lw=1,
            alpha=0.3,
            label="ROC fold %d (AUC = %0.2f)" % (index + 1, roc_auc),
        )

    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()


def get_confusion_matrix(true_vs_predicted: List[Dict[str, np.ndarray]]) -> None:
    ...


def plot_precision_recall_curve(true_vs_predicted: List[Dict[str, np.ndarray]]) -> None:

    plt.figure(2)

    cross_val_average_precision: List[float] = []
    cross_val_precision_recall: List[tuple] = []
    max_step_prec: int = 0
    max_step_recall: int = 0

    for index, record in enumerate(true_vs_predicted):

        average_precision = average_precision_score(record["true"], record["predicted"])
        cross_val_average_precision.append(average_precision)

        precision, recall, _ = precision_recall_curve(
            record["true"], record["predicted"]
        )
        cross_val_precision_recall.append((precision, recall))
        max_step_prec = (
            len(precision) if len(precision) > max_step_prec else max_step_prec
        )
        max_step_recall = (
            len(precision) if len(recall) > max_step_recall else max_step_recall
        )

        plt.plot(
            recall,
            precision,
            color="b",
            lw=1,
            alpha=0.3,
            label="PR fold %d (AP = %0.2f)" % (index + 1, average_precision),
        )
        plt.fill_between(recall, precision, color="b", alpha=0.01)

    mean_precision = np.mean(
        [
            fold_precision[0]
            for fold_precision in cross_val_precision_recall
            if len(fold_precision[0]) == max_step_prec
        ],
        axis=0,
    )
    mean_recall = np.mean(
        [
            fold_recall[1]
            for fold_recall in cross_val_precision_recall
            if len(fold_recall[0]) == max_step_recall
        ],
        axis=0,
    )

    mean_average_precision = sum(cross_val_average_precision) / len(
        cross_val_average_precision
    )
    plt.plot(
        mean_recall,
        mean_precision,
        linestyle="--",
        color="k",
        lw=1,
        label="Mean PR (mean AP = %0.2f)" % (mean_average_precision),
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        "Cross validation 2-class Precision-Recall curves: Mean AP={0:0.2f}".format(
            mean_average_precision
        )
    )

    plt.legend(loc="lower left")
    plt.show()


def calculate_binary_metrics(
    true_vs_predicted: List[Dict[str, np.ndarray]], label_to_binary: bool = True
) -> None:

    if label_to_binary:
        true_vs_predicted, unique_values = transform_label_to_binary(true_vs_predicted)

    # roc_and_auc_metrics(true_vs_predicted)
    plot_precision_recall_curve(true_vs_predicted)
