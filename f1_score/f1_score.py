# calcular_f1_lote.py
import os
import glob
import textwrap
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, confusion_matrix
)

# ================== CONFIGURAÇÃO FIXA ==================
INPUT_FOLDER = r"C:\Users\daniel\Desktop\LLM_MANUAL\LLM_FEW_SHOT\ANALISE_MANUAL_LLM_EW_SHOTS"
OUTPUT_FOLDER = r"C:\Users\daniel\Desktop\LLM_MANUAL\f1_score\f1"

# Gráficos
DPI = 300
MAX_LABEL_CHARS = 28
BAR_SORT_BY = "f1_macro"     # opções: "f1_macro", "f1_micro", "accuracy"
NORMALIZE_CONFUSION = True   # gera também a matriz normalizada
# =======================================================


def ensure_dirs(base_out: str) -> None:
    os.makedirs(os.path.join(base_out, "outputs"), exist_ok=True)
    os.makedirs(os.path.join(base_out, "figures"), exist_ok=True)
    os.makedirs(os.path.join(base_out, "figures", "per_file_conf_mat"), exist_ok=True)

def wrap_label(s, width=28):
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=True, replace_whitespace=False)) if s else s

def read_pairs(df: pd.DataFrame):
    y_true = df["sentimento_manual"].astype(str)
    y_pred = df["classificacao"].astype(str)
    m = min(len(y_true), len(y_pred))
    return y_true.iloc[:m].to_list(), y_pred.iloc[:m].to_list()

def plot_confusion(cm: np.ndarray, labels, title: str, outpath_png: str, outpath_svg: str, normalize: bool = False):
    import matplotlib.pyplot as plt  # importa só quando for usar

    cm_to_plot = cm.copy().astype(float)
    subtitle = ""
    if normalize and cm.sum() > 0:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_to_plot = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        subtitle = " (normalizada %)"

    fig, ax = plt.subplots(figsize=(6.6, 5.5), dpi=DPI)
    im = ax.imshow(cm_to_plot, aspect='auto', cmap="Blues")
    ax.set_title(title + subtitle)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([wrap_label(l, MAX_LABEL_CHARS) for l in labels], rotation=45, ha='right')
    ax.set_yticklabels([wrap_label(l, MAX_LABEL_CHARS) for l in labels])

    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            if normalize:
                ax.text(j, i, f"{cm_to_plot[i, j]*100:.1f}%", ha="center", va="center", fontsize=8)
            else:
                ax.text(j, i, f"{int(cm_to_plot[i, j])}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath_png, bbox_inches='tight', dpi=DPI)
    fig.savefig(outpath_svg, bbox_inches='tight', dpi=DPI)
    plt.close(fig)

def bars_metrics(df_metrics: pd.DataFrame, outpath_png: str, outpath_svg: str, sort_by: str = "f1_macro"):
    import matplotlib.pyplot as plt

    df_plot = df_metrics.copy()
    if sort_by in df_plot.columns:
        df_plot = df_plot.sort_values(sort_by, ascending=False).reset_index(drop=True)

    labels = [wrap_label(s, MAX_LABEL_CHARS) for s in df_plot["file"].tolist()]
    x = np.arange(len(labels))
    width = 0.26

    fig_h = max(5.0, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(max(9, len(labels)*0.6), fig_h), dpi=DPI)

    bars1 = ax.bar(x - width, df_plot["accuracy"], width, label="Accuracy")
    bars2 = ax.bar(x,          df_plot["f1_macro"], width, label="F1 Macro")
    bars3 = ax.bar(x + width,  df_plot["f1_micro"], width, label="F1 Micro")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Accuracy e F1 por arquivo (ordenado por {sort_by})")
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    def add_labels(bars):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.2f}", xy=(b.get_x()+b.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    for arr in (bars1, bars2, bars3):
        add_labels(arr)

    fig.tight_layout()
    fig.savefig(outpath_png, bbox_inches='tight', dpi=DPI)
    fig.savefig(outpath_svg, bbox_inches='tight', dpi=DPI)
    plt.close(fig)

def heatmap_per_class_f1(df_reports: pd.DataFrame, outpath_png: str, outpath_svg: str):
    import matplotlib.pyplot as plt
    if df_reports.empty:
        return

    pivot = df_reports.pivot_table(index="file", columns="class", values="f1-score")
    data = pivot.fillna(0.0).values
    classes = list(pivot.columns)
    files = [wrap_label(s, MAX_LABEL_CHARS) for s in list(pivot.index)]

    fig, ax = plt.subplots(figsize=(max(8, len(classes)*0.9), max(6, len(files)*0.45)), dpi=DPI)
    im = ax.imshow(data, aspect='auto', vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(files)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(files)
    ax.set_title("Heatmap de F1 por classe (arquivo x classe)")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if val < 0.5 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath_png, bbox_inches='tight', dpi=DPI)
    fig.savefig(outpath_svg, bbox_inches='tight', dpi=DPI)
    plt.close(fig)

def main():
    ensure_dirs(OUTPUT_FOLDER)

    files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*_analise.csv")))
    if not files:
        print("Nenhum arquivo *_analise.csv encontrado em", INPUT_FOLDER)
        return

    all_labels = set()
    y_true_all, y_pred_all = [], []
    metrics_rows = []
    report_rows = []

    for path in files:
        # leitura robusta
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

        if not {"classificacao", "sentimento_manual"}.issubset(df.columns):
            print(f"Ignorando (colunas ausentes): {os.path.basename(path)}")
            continue

        y_true, y_pred = read_pairs(df)
        if len(y_true) == 0:
            print(f"Ignorando (sem linhas válidas): {os.path.basename(path)}")
            continue

        all_labels.update(set(y_true) | set(y_pred))

        acc = accuracy_score(y_true, y_pred)
        f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_mic = f1_score(y_true, y_pred, average="micro", zero_division=0)

        base_name = os.path.basename(path).replace(" comments_analise.csv", "")
        metrics_rows.append({
            "file": base_name,
            "accuracy": acc,
            "f1_macro": f1_mac,
            "f1_micro": f1_mic,
            "n": len(y_true)
        })

        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        for cls, vals in rep.items():
            if cls in ["accuracy", "macro avg", "weighted avg"]:
                continue
            report_rows.append({
                "file": base_name,
                "class": str(cls),
                "precision": float(vals.get("precision", 0.0)),
                "recall": float(vals.get("recall", 0.0)),
                "f1-score": float(vals.get("f1-score", 0.0)),
                "support": int(vals.get("support", 0))
            })

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    # CSVs de saída
    df_metrics = pd.DataFrame(metrics_rows).sort_values("file").reset_index(drop=True)
    out_metrics = os.path.join(OUTPUT_FOLDER, "outputs", "metrics_by_file.csv")
    df_metrics.to_csv(out_metrics, index=False)

    df_reports = pd.DataFrame(report_rows)
    out_reports = os.path.join(OUTPUT_FOLDER, "outputs", "class_report_by_file.csv")
    df_reports.to_csv(out_reports, index=False)

    # Impressão das métricas por arquivo
    print("\n=== Métricas por arquivo ===")
    if not df_metrics.empty:
        print(df_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("Sem métricas (verifique se os CSVs possuem as colunas necessárias).")

    # Métricas globais
    labels_sorted = sorted(list(all_labels))
    cm_global = confusion_matrix(y_true_all, y_pred_all, labels=labels_sorted)

    f1_macro_global = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    f1_micro_global = f1_score(y_true_all, y_pred_all, average="micro", zero_division=0)
    acc_global = accuracy_score(y_true_all, y_pred_all)

    print("\n=== Resumo Global ===")
    print(f"Accuracy: {acc_global:.4f}")
    print(f"F1 Macro: {f1_macro_global:.4f}")
    print(f"F1 Micro: {f1_micro_global:.4f}")
    print(f"Arquivos processados: {len(df_metrics)}")
    print(f"Tabelas salvas em:\n - {out_metrics}\n - {out_reports}")

    # Gráficos
    figures_dir = os.path.join(OUTPUT_FOLDER, "figures")
    plot_confusion(
        cm_global, labels_sorted, "Matriz de Confusão Global",
        os.path.join(figures_dir, "global_confusion_matrix.png"),
        os.path.join(figures_dir, "global_confusion_matrix.svg"),
        normalize=False
    )
    if NORMALIZE_CONFUSION:
        plot_confusion(
            cm_global, labels_sorted, "Matriz de Confusão Global",
            os.path.join(figures_dir, "global_confusion_matrix_norm.png"),
            os.path.join(figures_dir, "global_confusion_matrix_norm.svg"),
            normalize=True
        )

    if not df_metrics.empty:
        bars_metrics(
            df_metrics,
            os.path.join(figures_dir, f"accuracy_f1_by_file_sorted_{BAR_SORT_BY}.png"),
            os.path.join(figures_dir, f"accuracy_f1_by_file_sorted_{BAR_SORT_BY}.svg"),
            sort_by=BAR_SORT_BY
        )

    if not df_reports.empty:
        heatmap_per_class_f1(
            df_reports,
            os.path.join(figures_dir, "per_class_f1_heatmap.png"),
            os.path.join(figures_dir, "per_class_f1_heatmap.svg")
        )

    print("Figuras salvas em:", figures_dir)

if __name__ == "__main__":
    main()
