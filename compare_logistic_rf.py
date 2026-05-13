import csv

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


DATASET_PATH = "all_findings_flat.csv"
RANDOM_STATE = 42

NUMERIC_COLUMNS = [
    "patch_lines",
    "patch_added",
    "patch_removed",
    "patch_files_touched",
    "patch_hunks",
    "patch_churn",
    "patch_net",
    "prompt_chars",
    "prompt_lines",
    "prompt_tokens",
    "temperature",
    "is_risky",
    "cwe_prevalence_overall",
    "cwe_severity_score",
    "cwe_weighted_severity",
]

BASE_FEATURES = [
    "patch_lines",
    "patch_added",
    "patch_removed",
    "patch_files_touched",
    "patch_hunks",
    "patch_churn",
    "patch_net",
    "prompt_chars",
    "prompt_lines",
    "prompt_tokens",
    "temperature",
]

OUTLIER_COLUMNS = [
    "patch_lines",
    "patch_added",
    "patch_removed",
    "patch_files_touched",
    "patch_hunks",
    "patch_churn",
    "patch_net",
    "prompt_chars",
    "prompt_lines",
    "prompt_tokens",
]

PLOT_COLUMNS = [
    "patch_lines",
    "patch_added",
    "patch_removed",
    "patch_files_touched",
    "patch_hunks",
    "patch_churn",
    "patch_net",
]

def load_numeric_data(path):
    rows = []

    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                rows.append([float(row[col]) for col in NUMERIC_COLUMNS])
            except (KeyError, ValueError):
                continue

    return np.array(rows, dtype=float)


def remove_outliers(data, column_index):
    values = data[:, column_index]
    q1 = np.quantile(values, 0.25)
    q3 = np.quantile(values, 0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return data[(values >= lower) & (values <= upper)]


def prompt_size_category(prompt_chars):
    # Equivalent to pd.cut(..., bins=[0, 500, 1000, 2000, inf], labels=[0, 1, 2, 3]).
    return np.digitize(prompt_chars, [500, 1000, 2000], right=True).astype(float)


def build_dataset():
    raw_data = load_numeric_data(DATASET_PATH)
    data = raw_data.copy()
    col_idx = {name: idx for idx, name in enumerate(NUMERIC_COLUMNS)}

    data = data[data[:, col_idx["patch_lines"]] > 0]

    for column in OUTLIER_COLUMNS:
        data = remove_outliers(data, col_idx[column])

    base_features = np.column_stack([data[:, col_idx[col]] for col in BASE_FEATURES])

    patch_lines = data[:, col_idx["patch_lines"]]
    patch_added = data[:, col_idx["patch_added"]]
    patch_removed = data[:, col_idx["patch_removed"]]
    patch_files_touched = data[:, col_idx["patch_files_touched"]]
    patch_hunks = data[:, col_idx["patch_hunks"]]
    patch_churn = data[:, col_idx["patch_churn"]]
    patch_net = data[:, col_idx["patch_net"]]
    prompt_chars = data[:, col_idx["prompt_chars"]]
    prompt_lines = data[:, col_idx["prompt_lines"]]
    prompt_tokens = data[:, col_idx["prompt_tokens"]]
    temperature = data[:, col_idx["temperature"]]

    derived_features = np.column_stack(
        [
            patch_churn / (patch_lines + 1),
            patch_added / (patch_removed + 1),
            patch_net / (patch_lines + 1),
            patch_hunks / (patch_files_touched + 1),
            prompt_chars / (prompt_lines + 1),
            prompt_tokens / (prompt_chars + 1),
            prompt_size_category(prompt_chars),
            patch_hunks * patch_files_touched,
            patch_churn / (patch_files_touched + 1),
            temperature * prompt_chars,
            temperature * patch_lines,
        ]
    )

    feature_names = BASE_FEATURES + [
        "patch_density",
        "add_remove_ratio",
        "net_per_line",
        "hunks_per_file",
        "prompt_density",
        "prompt_token_density",
        "prompt_size_category",
        "patch_complexity",
        "change_intensity",
        "temp_x_prompt_size",
        "temp_x_patch_size",
    ]

    x = np.column_stack([base_features, derived_features]).astype(float)
    y = data[:, col_idx["is_risky"]].astype(int)

    x = MinMaxScaler().fit_transform(x)

    return x, y, feature_names, raw_data, col_idx


def plot_variable_distributions(data, col_idx, output_path, split_by_risk=False):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    risk_values = data[:, col_idx["is_risky"]].astype(int)
    axis_scales = {}

    # Mantem o mesmo teto de Y entre os dois graficos (geral e por risco).
    for column in PLOT_COLUMNS:
        values = data[:, col_idx[column]]
        hist_counts, _ = np.histogram(values, bins=30)
        axis_scales[column] = {
            "ylim": (0, hist_counts.max() * 1.05 if hist_counts.size and hist_counts.max() > 0 else 1),
        }

    for idx, column in enumerate(PLOT_COLUMNS):
        ax = axes[idx]
        values = data[:, col_idx[column]]

        if split_by_risk:
            safe_values = values[risk_values == 0]
            risky_values = values[risk_values == 1]

            sns.histplot(
                safe_values,
                kde=True,
                bins=30,
                color="green",
                alpha=0.45,
                stat="count",
                ax=ax,
                label="Sem risco",
            )
            sns.histplot(
                risky_values,
                kde=True,
                bins=30,
                color="red",
                alpha=0.45,
                stat="count",
                ax=ax,
                label="Com risco",
            )
            ax.legend(fontsize=8)
        else:
            sns.histplot(
                values,
                kde=True,
                bins=30,
                color="#4c72b0",
                ax=ax,
            )

        ax.set_ylim(axis_scales[column]["ylim"])
        ax.set_title(f"Distribuicao de {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")

    for empty_ax in axes[len(PLOT_COLUMNS) :]:
        empty_ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(name, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    else:
        y_score = y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_risk": precision_score(y_test, y_pred, zero_division=0),
        "recall_risk": recall_score(y_test, y_pred, zero_division=0),
        "f1_risk": f1_score(y_test, y_pred, zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
    }

    return {
        "name": name,
        "model": model,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, digits=3),
    }


def print_result(result):
    print(f"\n{'=' * 72}")
    print(result["name"])
    print(f"{'=' * 72}")

    for metric, value in result["metrics"].items():
        print(f"{metric:20s}: {value:.6f}")

    print("\nMatriz de confusao:")
    print(result["confusion_matrix"])
    print("\nRelatorio de classificacao:")
    print(result["classification_report"])


def main():
    x, y, feature_names, data_for_plots, col_idx = build_dataset()

    plot_variable_distributions(data_for_plots, col_idx, "distribuicao_variaveis.png")
    plot_variable_distributions(
        data_for_plots,
        col_idx,
        "distribuicao_variaveis_por_risco.png",
        split_by_risk=True,
    )

    print(f"Amostras apos pre-processamento: {len(y)}")
    print(f"Total de features: {len(feature_names)}")
    print(f"Classe 0 (seguro): {(y == 0).sum()}")
    print(f"Classe 1 (risco):  {(y == 1).sum()}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = [
        (
            "Regressao Logistica",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]

    results = [
        evaluate_model(name, model, x_train, x_test, y_train, y_test)
        for name, model in models
    ]

    for result in results:
        print_result(result)

    best_by_f1 = max(results, key=lambda result: result["metrics"]["f1_risk"])
    best_by_auc = max(results, key=lambda result: result["metrics"]["roc_auc"])

    print(f"\n{'=' * 72}")
    print("Comparacao final")
    print(f"{'=' * 72}")
    print(f"Melhor por F1 da classe de risco: {best_by_f1['name']}")
    print(f"Melhor por ROC-AUC: {best_by_auc['name']}")


if __name__ == "__main__":
    main()
