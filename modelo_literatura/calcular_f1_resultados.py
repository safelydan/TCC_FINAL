
# -*- coding: utf-8 -*-
import os
import glob
import textwrap
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, confusion_matrix
)

# ================== CONFIGURAÇÕES ==================
INPUT_FOLDER  = r"C:\Users\daniel\Desktop\LLM_FINAL\modelo_literatura\resultados"   # pasta onde extraímos os CSVs
OUTPUT_FOLDER = r"C:\Users\daniel\Desktop\LLM_FINAL\modelo_literatura\f1_resultados"

DPI = 300
MAX_LABEL_CHARS = 28
BAR_SORT_BY = "f1_macro"
NORMALIZE_CONFUSION = True

# Estilo básico (sem cores customizadas para manter compatibilidade)
GRID_STYLE = dict(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
BAR_EDGECOLOR = "#222222"
BAR_LINEWIDTH = 0.6
TITLE_FONTSIZE = 12
LABEL_FONTSIZE = 10
TICK_FONTSIZE  = 9
ANNOT_FONTSIZE = 8

# ================== UTILS ==================
def ensure_dirs(base_out: str) -> None:
    os.makedirs(os.path.join(base_out, "outputs"), exist_ok=True)
    os.makedirs(os.path.join(base_out, "figures"), exist_ok=True)
    os.makedirs(os.path.join(base_out, "figures", "per_file_conf_mat"), exist_ok=True)

def wrap_label(s, width=28):
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=True, replace_whitespace=False)) if s else s

def normalize_label(x):
    """
    Normaliza rótulos para o conjunto {'Positive','Neutral','Negative'}.
    Aceita inteiros (1,0,-1) e strings em várias capitalizações.
    """
    if pd.isna(x):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        xi = int(x)
        return {1: "Positive", 0: "Neutral", -1: "Negative"}.get(xi, str(x))
    s = str(x).strip().lower()
    mapping = {
        "1": "Positive", "0": "Neutral", "-1": "Negative",
        "positive": "Positive", "pos": "Positive", "positivo": "Positive",
        "neutral": "Neutral", "neutro": "Neutral",
        "negative": "Negative", "neg": "Negative", "negativo": "Negative",
        "mixed": "Neutral"  # regra pedida antes
    }
    return mapping.get(s, s.title())

def pick_columns(df: pd.DataFrame):
    """
    Tenta descobrir automaticamente qual coluna é o y_true (manual) e qual é o y_pred (modelo).
    """
    cand_true = ["sentimento_manual", "manual", "rotulo_manual", "y_true", "true", "gold"]
    cand_pred = ["classificacao", "sentiment", "predicao", "pred", "y_pred", "modelo", "label"]
    y_true_col = next((c for c in cand_true if c in df.columns), None)
    y_pred_col = next((c for c in cand_pred if c in df.columns), None)
    return y_true_col, y_pred_col

def read_pairs(df: pd.DataFrame):
    y_true_col, y_pred_col = pick_columns(df)
    if not y_true_col or not y_pred_col:
        return [], [], y_true_col, y_pred_col
    y_true = df[y_true_col].map(normalize_label)
    y_pred = df[y_pred_col].map(normalize_label)
    m = min(len(y_true), len(y_pred))
    return y_true.iloc[:m].to_list(), y_pred.iloc[:m].to_list(), y_true_col, y_pred_col

# ================== PLOTS ==================
def plot_confusion(cm: np.ndarray, labels, title: str, outpath_png: str, outpath_svg: str, normalize: bool = False):
    import matplotlib.pyplot as plt
    cm_to_plot = cm.copy().astype(float)
    subtitle = ""
    if normalize and cm.sum() > 0:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_to_plot = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        subtitle = " (normalizada %)"
    fig, ax = plt.subplots(figsize=(6.6, 5.5), dpi=DPI)
    im = ax.imshow(cm_to_plot, aspect='auto')  # sem cmap explícito
    ax.set_title(title + subtitle, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Predito", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Verdadeiro", fontsize=LABEL_FONTSIZE)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([wrap_label(l, MAX_LABEL_CHARS) for l in labels], rotation=45, ha='right', fontsize=TICK_FONTSIZE)
    ax.set_yticklabels([wrap_label(l, MAX_LABEL_CHARS) for l in labels], fontsize=TICK_FONTSIZE)
    vmax = np.max(cm_to_plot) if cm_to_plot.size else 1.0
    thr = 0.6 * vmax if vmax > 0 else 0.0
    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            val = cm_to_plot[i, j]
            txt = f"{val*100:.1f}%" if normalize else f"{int(val)}"
            color = "black" if val >= thr else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=ANNOT_FONTSIZE, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(outpath_png, bbox_inches='tight', dpi=DPI)
    fig.savefig(outpath_svg, bbox_inches='tight', dpi=DPI)
    plt.close(fig)

def bars_metrics(df_metrics: pd.DataFrame, outpath_png: str, outpath_svg: str, sort_by: str = "f1_macro"):
    import matplotlib.pyplot as plt
    if df_metrics.empty:
        return
    df_plot = df_metrics.copy()
    if sort_by in df_plot.columns:
        df_plot = df_plot.sort_values(sort_by, ascending=False).reset_index(drop=True)
    labels = [wrap_label(s, MAX_LABEL_CHARS) for s in df_plot["file"].tolist()]
    x = np.arange(len(labels))
    width = 0.26
    fig_h = max(5.0, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(max(9, len(labels)*0.6), fig_h), dpi=DPI)
    bars1 = ax.bar(x - width, df_plot["accuracy"], width, label="Accuracy", edgecolor=BAR_EDGECOLOR, linewidth=BAR_LINEWIDTH)
    bars2 = ax.bar(x,          df_plot["f1_macro"], width, label="F1 Macro", edgecolor=BAR_EDGECOLOR, linewidth=BAR_LINEWIDTH)
    bars3 = ax.bar(x + width,  df_plot["f1_micro"], width, label="F1 Micro", edgecolor=BAR_EDGECOLOR, linewidth=BAR_LINEWIDTH)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=TICK_FONTSIZE)
    ax.set_ylabel("Score", fontsize=LABEL_FONTSIZE)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Accuracy e F1 por arquivo (ordenado por {sort_by})", fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=TICK_FONTSIZE)
    ax.grid(**GRID_STYLE)
    def add_labels(bars):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.2f}", xy=(b.get_x()+b.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=ANNOT_FONTSIZE, color="#111111")
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
    im = ax.imshow(data, aspect='auto', vmin=0, vmax=1)  # sem cmap explícito
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(files)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=TICK_FONTSIZE)
    ax.set_yticklabels(files, fontsize=TICK_FONTSIZE)
    ax.set_title("Heatmap de F1 por classe (arquivo x classe)", fontsize=TITLE_FONTSIZE)
    thr = 0.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "black" if val >= thr else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=ANNOT_FONTSIZE, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(outpath_png, bbox_inches='tight', dpi=DPI)
    fig.savefig(outpath_svg, bbox_inches='tight', dpi=DPI)
    plt.close(fig)

# ================== MAIN ==================
def main():
    ensure_dirs(OUTPUT_FOLDER)
    files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "**", "*comments_*.csv"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "**", "*.csv"), recursive=True))

    if not files:
        print("Nenhum CSV encontrado em", INPUT_FOLDER)
        return

    all_labels = set()
    y_true_all, y_pred_all = [], []
    metrics_rows = []
    report_rows = []

    for path in files:
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

        y_true, y_pred, y_true_col, y_pred_col = read_pairs(df)
        if not y_true_col or not y_pred_col:
            print(f"Ignorando (não conseguiu identificar colunas): {os.path.basename(path)}")
            continue

        if len(y_true) == 0:
            print(f"Ignorando (sem linhas válidas): {os.path.basename(path)}")
            continue

        all_labels.update(set(y_true) | set(y_pred))

        acc = accuracy_score(y_true, y_pred)
        f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_mic = f1_score(y_true, y_pred, average="micro", zero_division=0)

        base_name = os.path.basename(path).replace(".csv", "")
        metrics_rows.append({
            "file": base_name,
            "accuracy": acc,
            "f1_macro": f1_mac,
            "f1_micro": f1_mic,
            "n": len(y_true),
            "y_true_col": y_true_col,
            "y_pred_col": y_pred_col,
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

    df_metrics = pd.DataFrame(metrics_rows).sort_values("file").reset_index(drop=True)
    out_metrics = os.path.join(OUTPUT_FOLDER, "outputs", "metrics_by_file.csv")
    df_metrics.to_csv(out_metrics, index=False)

    df_reports = pd.DataFrame(report_rows)
    out_reports = os.path.join(OUTPUT_FOLDER, "outputs", "class_report_by_file.csv")
    df_reports.to_csv(out_reports, index=False)

    print("\n=== Métricas por arquivo ===")
    if not df_metrics.empty:
        print(df_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else f"{x}"))
    else:
        print("Sem métricas (verifique se os CSVs possuem as colunas necessárias).")

    labels_sorted = sorted(list(all_labels))
    if y_true_all and y_pred_all:
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

        figures_dir = os.path.join(OUTPUT_FOLDER, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        # Matriz de confusão (absoluta e normalizada)
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

        # Barras por arquivo
        if not df_metrics.empty:
            bars_metrics(
                df_metrics,
                os.path.join(figures_dir, f"accuracy_f1_by_file_sorted_{BAR_SORT_BY}.png"),
                os.path.join(figures_dir, f"accuracy_f1_by_file_sorted_{BAR_SORT_BY}.svg"),
                sort_by=BAR_SORT_BY
            )

        # Heatmap por classe
        if not df_reports.empty:
            heatmap_per_class_f1(
                df_reports,
                os.path.join(figures_dir, "per_class_f1_heatmap.png"),
                os.path.join(figures_dir, "per_class_f1_heatmap.svg")
            )

        print("Figuras salvas em:", figures_dir)
    else:
        print("Não foi possível calcular métricas globais (listas vazias).")

if __name__ == "__main__":
    main()
