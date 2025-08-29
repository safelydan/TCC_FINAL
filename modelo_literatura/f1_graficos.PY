import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

INPUT_FOLDER  = r"C:\Users\daniel\Desktop\LLM_FINAL\modelo_literatura\resultados"   # pasta onde extraímos os CSVs
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "_reports")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

LABELS = ["Negative", "Neutral", "Positive"]

# ======================
# Funções auxiliares
# ======================

def plot_f1_bars(report_dict, file_name):
    """Gera gráfico de barras dos F1-scores por classe."""
    f1s = [report_dict[label]["f1-score"] for label in LABELS]
    plt.figure(figsize=(6,4))
    sns.barplot(x=LABELS, y=f1s)
    plt.ylim(0,1)
    plt.title(f"F1-score por classe\n{file_name}")
    plt.ylabel("F1-score")
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{file_name}_f1.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(cm, file_name):
    """Gera heatmap da matriz de confusão normalizada."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_norm, annot=True, cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, fmt=".2f")
    plt.title(f"Matriz de Confusão Normalizada\n{file_name}")
    plt.ylabel("Verdadeiro")
    plt.xlabel("Previsto")
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{file_name}_cm.png"), dpi=300, bbox_inches="tight")
    plt.close()

# ======================
# Loop pelos arquivos
# ======================
rows = []
for path in glob.glob(os.path.join(INPUT_FOLDER, "*scored.csv")):
    df = pd.read_csv(path)
    base = os.path.splitext(os.path.basename(path))[0]

    y_true = df["sentimento_manual"]
    y_pred = df["sentiment"]

    # classification_report
    rep = classification_report(y_true, y_pred, labels=LABELS, target_names=LABELS,
                                digits=2, output_dict=True, zero_division=0)
    
    # salva gráfico de barras de F1
    plot_f1_bars(rep, base)

    # matriz de confusão
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plot_confusion_matrix(cm, base)

    # adiciona ao resumo
    rows.append({
        "file": base,
        "acc": rep["accuracy"],
        "f1_neg": rep["Negative"]["f1-score"],
        "f1_neu": rep["Neutral"]["f1-score"],
        "f1_pos": rep["Positive"]["f1-score"],
        "f1_macro": rep["macro avg"]["f1-score"],
        "f1_weighted": rep["weighted avg"]["f1-score"],
    })

# ======================
# Gráfico agregado
# ======================
summary = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(data=summary, x="f1_macro", y="file", palette="viridis")
plt.xlim(0,1)
plt.title("Comparação de F1-macro entre arquivos")
plt.xlabel("F1-macro")
plt.ylabel("Arquivo")
plt.savefig(os.path.join(OUTPUT_FOLDER, "_comparacao_f1_macro.png"), dpi=300, bbox_inches="tight")
plt.close()

# salva resumo em CSV
summary.to_csv(os.path.join(OUTPUT_FOLDER, "_summary_with_f1.csv"), index=False, encoding="utf-8")
print("Gráficos e resumo salvos em:", OUTPUT_FOLDER)
