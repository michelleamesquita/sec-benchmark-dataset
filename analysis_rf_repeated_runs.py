"""
Repeticao controlada do treinamento (RF + Regressao Logistica) com o mesmo
pipeline de dados de compare_logistic_rf.py (tabela de metricas), N splits/treinos
com sementes derivadas deterministicamente de RANDOM_SEED.

- RANDOM_SEED: constante no topo, nao reatribuida.
- N runs: sementes geradas por np.random.default_rng(RANDOM_SEED).
- Hiperparametros fixos (iguais a compare_logistic_rf.py).
- Saida: CSV, resumo media +/- desvio, graficos (boxplot, barras, e figuras do
  run de referencia seed=42: metricas por classe, matriz de confusao, importancia,
  SHAP — alinhados ao analysis_final.py).
"""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from compare_logistic_rf import build_dataset

try:
    import shap

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

# -----------------------------------------------------------------------------
# Semente base: definida uma unica vez no topo (nao e reatribuida no script).
# -----------------------------------------------------------------------------
RANDOM_SEED = 42

TEST_SIZE = 0.2

# Mesmos hiperparametros que compare_logistic_rf.py
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 15
RF_MIN_SAMPLES_LEAF = 1
RF_MAX_FEATURES = "sqrt"
RF_CLASS_WEIGHT = "balanced"

LR_MAX_ITER = 1000
LR_CLASS_WEIGHT = "balanced"

METRIC_LABELS = {
    "accuracy": "Acuracia",
    "precision_risk": "Precisao (classe 1)",
    "recall_risk": "Revocacao (classe 1)",
    "f1_risk": "F1 (classe 1)",
    "roc_auc": "ROC-AUC",
}


def _metrics_binary(y_true, y_pred, y_score) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_risk": precision_score(y_true, y_pred, zero_division=0),
        "recall_risk": recall_score(y_true, y_pred, zero_division=0),
        "f1_risk": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def evaluate_both_models(
    x: np.ndarray,
    y: np.ndarray,
    run_seed: int,
) -> dict[str, float]:
    """Mesmo split para os dois modelos; semente controla split + RF (+ LR)."""
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=run_seed,
    )

    lr = LogisticRegression(
        class_weight=LR_CLASS_WEIGHT,
        max_iter=LR_MAX_ITER,
        random_state=run_seed,
    )
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)
    y_score_lr = lr.predict_proba(x_test)[:, 1]
    lr_m = _metrics_binary(y_test, y_pred_lr, y_score_lr)

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        class_weight=RF_CLASS_WEIGHT,
        random_state=run_seed,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    y_score_rf = rf.predict_proba(x_test)[:, 1]
    rf_m = _metrics_binary(y_test, y_pred_rf, y_score_rf)

    out = {f"lr_{k}": v for k, v in lr_m.items()}
    out.update({f"rf_{k}": v for k, v in rf_m.items()})
    return out


def _plot_boxplots(df_long: pd.DataFrame, path: str) -> None:
    metrics_order = list(METRIC_LABELS.keys())
    df_long = df_long[df_long["metric"].isin(metrics_order)]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, m in zip(axes, metrics_order):
        sub = df_long[df_long["metric"] == m]
        order = ["Regressao Logistica", "Random Forest"]
        sns.boxplot(
            data=sub,
            x="modelo",
            y="valor",
            order=order,
            palette=["#4c72b0", "#dd8452"],
            ax=ax,
        )
        ax.set_title(METRIC_LABELS[m])
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle(
        "Distribuicao das metricas ao longo de N execucoes (sementes derivadas de RANDOM_SEED)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _fit_rf_reference(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    seed: int,
) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Treina RF com o mesmo split/seed da referencia (ex.: tabela LaTeX)."""
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=seed,
    )
    cols = list(feature_names)
    x_test_df = pd.DataFrame(x_test, columns=cols)

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        class_weight=RF_CLASS_WEIGHT,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)
    return rf, y_train, y_test, x_test_df, pd.DataFrame(x_train, columns=cols)


def save_reference_rf_figures(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    plot_prefix: str,
    seed: int = RANDOM_SEED,
) -> None:
    """
    Figuras do mesmo estilo que analysis_final.py (run unico, seed fixa):
    metricas por classe, matriz de confusao, importancia de features, SHAP.
    """
    rf, _, y_test, x_test_df, _ = _fit_rf_reference(
        x, y, feature_names, seed
    )
    y_pred = rf.predict(x_test_df.values)

    # --- Metricas por classe (estilo regressao.png) ---
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    classes = ["0", "1"]
    metrics = ["precision", "recall", "f1-score"]
    plot_df = pd.DataFrame(
        {m: [report_dict.get(c, {}).get(m, 0) for c in classes] for m in metrics},
        index=["Classe 0 (Sem risco)", "Classe 1 (Com risco)"],
    )
    plot_df.plot(kind="bar", figsize=(7, 5))
    plt.ylim(0, 1)
    plt.title("Métricas por Classe - Random Forest")
    plt.ylabel("Score")
    plt.xlabel("Classe")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}_metricas_classe.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Matriz de confusao (estilo previsao.png) ---
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = (cm / cm.sum()) * 100
    cm_annot = np.array(
        [
            [f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ]
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=cm_annot,
        fmt="",
        cmap="Blues",
        annot_kws={"fontsize": 11},
        xticklabels=["Previsto 0 (Sem risco)", "Previsto 1 (Com risco)"],
        yticklabels=["Real 0 (Sem risco)", "Real 1 (Com risco)"],
    )
    plt.title("Matriz de Confusão - Random Forest", fontsize=13)
    plt.xlabel("Previsto", fontsize=12)
    plt.ylabel("Real", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}_matriz_confusao.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Importancia de features (estilo coeficiente.png) ---
    feat_importance = (
        pd.DataFrame({"Feature": list(feature_names), "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=False)
    )
    top_k = min(15, len(feat_importance))
    plt.figure(figsize=(9, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=feat_importance.head(top_k),
    )
    plt.title("Top Features - Importância (Random Forest)")
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- SHAP (estilo shap_beeswarm.png) ---
    if not _HAS_SHAP:
        print(
            "\nAviso: pacote 'shap' nao instalado; pulando grafico SHAP. "
            "Instale com: pip install shap",
            file=sys.stderr,
        )
        return

    n_sample = min(200, len(x_test_df))
    x_sample = x_test_df.sample(n=n_sample, random_state=seed)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(x_sample)
    if isinstance(shap_values, list):
        shap_values_c1 = shap_values[1]
    elif len(np.asarray(shap_values).shape) == 3:
        shap_values_c1 = np.asarray(shap_values)[:, :, 1]
    else:
        shap_values_c1 = shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_c1,
        x_sample,
        show=False,
        max_display=15,
    )
    plt.title(
        "SHAP: Impacto das features (predição de risco)",
        fontsize=14,
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_mean_std(df_runs: pd.DataFrame, path: str) -> None:
    metric_cols = list(METRIC_LABELS.keys())
    lr_means = [df_runs[f"lr_{m}"].mean() for m in metric_cols]
    lr_stds = [df_runs[f"lr_{m}"].std(ddof=1) for m in metric_cols]
    rf_means = [df_runs[f"rf_{m}"].mean() for m in metric_cols]
    rf_stds = [df_runs[f"rf_{m}"].std(ddof=1) for m in metric_cols]

    x_pos = np.arange(len(metric_cols))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        x_pos - width / 2,
        lr_means,
        width,
        yerr=lr_stds,
        capsize=4,
        label="Regressao Logistica",
        color="#4c72b0",
    )
    ax.bar(
        x_pos + width / 2,
        rf_means,
        width,
        yerr=rf_stds,
        capsize=4,
        label="Random Forest",
        color="#dd8452",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metric_cols], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor")
    ax.legend()
    ax.set_title("Media +/- desvio-padrao entre N runs (hiperparametros fixos)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="N runs RF+RL (pipeline compare_logistic_rf), graficos e CSV."
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=30,
        help="Numero de runs (sementes distintas derivadas de RANDOM_SEED).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="rf_lr_repeated_runs_metrics.csv",
        help="CSV com uma linha por run.",
    )
    parser.add_argument(
        "--plot-prefix",
        type=str,
        default="rf_lr_repeated_runs",
        help="Prefixo dos arquivos PNG (boxplot, barras, figuras RF referencia).",
    )
    parser.add_argument(
        "--skip-reference-figures",
        action="store_true",
        help="Nao gera metricas por classe, matriz de confusao, importancia e SHAP.",
    )
    args = parser.parse_args()

    if args.n_repeats < 2:
        print("Use --n-repeats >= 2 para desvio-padrao.", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(RANDOM_SEED)
    run_seeds = rng.integers(low=0, high=2**31 - 1, size=args.n_repeats, dtype=np.int64)

    print(f"RANDOM_SEED (constante no topo): {RANDOM_SEED} (nao reatribuida no script)")
    print("Pipeline de dados: compare_logistic_rf.build_dataset()")
    print(f"N repeats: {args.n_repeats}")
    print(
        "RF:",
        f"n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH}, "
        f"min_samples_leaf={RF_MIN_SAMPLES_LEAF}, max_features={RF_MAX_FEATURES!r}, "
        f"class_weight={RF_CLASS_WEIGHT!r}",
    )
    print(
        "RL:",
        f"class_weight={LR_CLASS_WEIGHT!r}, max_iter={LR_MAX_ITER}",
    )

    x, y, feature_names, _, _ = build_dataset()
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    print(f"Amostras: {len(y)} | Features: {x.shape[1]} ({len(feature_names)} nomes)")

    rows = []
    for i, run_seed in enumerate(run_seeds):
        run_seed = int(run_seed)
        metrics = evaluate_both_models(x, y, run_seed)
        metrics["run_index"] = i
        metrics["run_seed"] = run_seed
        rows.append(metrics)
        step = max(1, args.n_repeats // 10)
        if (i + 1) % step == 0 or i == 0:
            print(f"  Run {i + 1}/{args.n_repeats} (seed={run_seed}) OK")

    df_runs = pd.DataFrame(rows)
    df_runs.to_csv(args.output_csv, index=False)
    print(f"\nCSV: {args.output_csv}")

    # Referencia identica a tabela (um unico run com seed 42 em split + modelos)
    ref = evaluate_both_models(x, y, RANDOM_SEED)
    print("\n--- Referencia (1 run, seed=42 no split e nos modelos) ---")
    print("Regressao Logistica | acc  prec(1)  rec(1)  f1(1)  roc_auc")
    print(
        f"                    | {ref['lr_accuracy']:.3f}  {ref['lr_precision_risk']:.3f}  "
        f"{ref['lr_recall_risk']:.3f}  {ref['lr_f1_risk']:.3f}  {ref['lr_roc_auc']:.3f}"
    )
    print("Random Forest      | acc  prec(1)  rec(1)  f1(1)  roc_auc")
    print(
        f"                    | {ref['rf_accuracy']:.3f}  {ref['rf_precision_risk']:.3f}  "
        f"{ref['rf_recall_risk']:.3f}  {ref['rf_f1_risk']:.3f}  {ref['rf_roc_auc']:.3f}"
    )
    print("(Valores acima devem coincidir com compare_logistic_rf.py / sua tabela LaTeX.)")

    metric_cols = list(METRIC_LABELS.keys())
    print("\n--- Media +/- desvio (N runs, sementes aleatorias derivadas) ---")
    for col in metric_cols:
        for prefix, name in [("lr_", "RL"), ("rf_", "RF")]:
            mean_v = df_runs[f"{prefix}{col}"].mean()
            std_v = df_runs[f"{prefix}{col}"].std(ddof=1)
            print(f"  {name} {col:18s}: {mean_v:.6f} +/- {std_v:.6f}")

    # Long format para boxplot
    long_rows = []
    for _, row in df_runs.iterrows():
        for col in metric_cols:
            long_rows.append(
                {
                    "run_index": row["run_index"],
                    "metric": col,
                    "modelo": "Regressao Logistica",
                    "valor": row[f"lr_{col}"],
                }
            )
            long_rows.append(
                {
                    "run_index": row["run_index"],
                    "metric": col,
                    "modelo": "Random Forest",
                    "valor": row[f"rf_{col}"],
                }
            )
    df_long = pd.DataFrame(long_rows)

    box_path = f"{args.plot_prefix}_boxplot.png"
    bar_path = f"{args.plot_prefix}_mean_std.png"
    _plot_boxplots(df_long, box_path)
    _plot_mean_std(df_runs, bar_path)
    print(f"\nGraficos: {box_path} , {bar_path}")

    if not args.skip_reference_figures:
        print(
            f"\nFiguras do run de referencia (seed={RANDOM_SEED}, RF, estilo analysis_final)..."
        )
        save_reference_rf_figures(x, y, feature_names, args.plot_prefix, RANDOM_SEED)
        print(f"  {args.plot_prefix}_metricas_classe.png")
        print(f"  {args.plot_prefix}_matriz_confusao.png")
        print(f"  {args.plot_prefix}_feature_importance.png")
        if _HAS_SHAP:
            print(f"  {args.plot_prefix}_shap_beeswarm.png")


if __name__ == "__main__":
    main()
