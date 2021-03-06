import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_curve,
)


def calc_chi_2_test(df, col_nm):

    c, p, dof, expected = chi2_contingency(pd.crosstab(df[col_nm], df["subscriber"]))
    print(f"The p-value is {round(p, 4)}")

    if p > 0.05:
        print(
            f"The result is not significant at the 95% confidence level. "
            f"The variable {col_nm} does not influence subscriber rates"
        )
    else:
        print(
            f"The result is significant at the 95% confidence level. "
            f"The variable {col_nm} influences subscriber rates"
        )


def _get_metrics(true_labels, predicted_labels, prob_labels):
    print("Log Loss:", np.round(log_loss(true_labels, prob_labels), 4))
    print("Accuracy:", np.round(accuracy_score(true_labels, predicted_labels), 4))
    print(
        "Precision:",
        np.round(
            precision_score(true_labels, predicted_labels, average="weighted"), 4,
        ),
    )
    print(
        "Recall:",
        np.round(recall_score(true_labels, predicted_labels, average="weighted"), 4),
    )
    print(
        "F1 Score:",
        np.round(f1_score(true_labels, predicted_labels, average="weighted"), 4),
    )


def _display_confusion_matrix(true_labels, predicted_labels, classes=(1, 0)):
    total_classes = len(classes)
    level_labels = [total_classes * [0], list(range(total_classes))]

    cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels, labels=classes)
    cm_frame = pd.DataFrame(
        data=cm,
        columns=pd.MultiIndex(levels=[["Predicted:"], classes], codes=level_labels),
        index=pd.MultiIndex(levels=[["Actual:"], classes], codes=level_labels),
    )
    print(cm_frame)


def _display_classification_report(true_labels, predicted_labels, classes=(1, 0)):
    report = classification_report(
        y_true=true_labels, y_pred=predicted_labels, labels=classes
    )
    print(report)


def display_model_performance_metrics(
    true_labels, predicted_labels, prob_labels, classes=(1, 0)
):
    print("Model Performance metrics:")
    print("-" * 30)
    _get_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        prob_labels=prob_labels,
    )
    print("\nModel Classification report:")
    print("-" * 30)
    _display_classification_report(
        true_labels=true_labels, predicted_labels=predicted_labels, classes=classes
    )
    print("\nPrediction Confusion Matrix:")
    print("-" * 30)
    _display_confusion_matrix(
        true_labels=true_labels, predicted_labels=predicted_labels, classes=classes
    )


def plot_model_roc_curve(true_labels, predicted_prob_labels):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        true_labels, predicted_prob_labels
    )
    roc_auc = auc(false_positive_rate, true_positive_rate)

    print(f"The AUC Value is {round(roc_auc, 2)}")

    plt.figure(figsize=(10, 10))
    plt.title("Receiver Operating Characteristic")
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        color="green",
        label=f"AUC = {round(roc_auc, 2)}",
    )
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.axis("tight")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")


def run_evaluate_model(df):

    assert "label" in df, "Evaluation dataframe is missing column label"
    assert "pred" in df, "Evaluation dataframe is missing column pred"
    assert "pred_prob" in df, "Evaluation dataframe is missing column pred_prob"

    display_model_performance_metrics(df["label"], df["pred"], df["pred_prob"])
    plot_model_roc_curve(df["label"], df["pred_prob"])
