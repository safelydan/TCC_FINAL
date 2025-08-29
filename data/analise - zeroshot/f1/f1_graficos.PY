import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

INPUT_FOLDER  = r"C:\Users\daniel\Desktop\LLM_FINAL\data\analise - zeroshot"
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "graficos")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

LABELS = ["Negative", "Neutral", "Positive"]

def plot_f1_bars(report_dict, file_name):

    f1s = [
        report_dict.get(label, {}).get("f1-score", 0.0)
        for label in LABELS
    ]
    plt.figure(figsize=(6,4))
    sns.barplot(x=LABELS, y=f1s)  
    plt.ylim(0,1)
    plt.title(f"F1-score por classe\n{file_name}")
    plt.ylabel("F1-score")
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{file_name}_f1.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(cm, file_name):
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums!=0)
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm_norm, annot=True, cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, fmt=".2f")
    plt.title(f"Matriz de Confusão Normalizada\n{file_name}")
    plt.ylabel("Verdadeiro")
    plt.xlabel("Previsto")
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{file_name}_cm.png"), dpi=300, bbox_inches="tight")
    plt.close()

rows = []
for path in glob.glob(os.path.join(INPUT_FOLDER, "*.csv")):
    df = pd.read_csv(path)
    base = os.path.splitext(os.path.basename(path))[0]


    y_true = df["sentimento_manual"].astype(str)
    y_pred = df["classificacao"].astype(str)


    rep = classification_report(
        y_true, y_pred,
        labels=LABELS, target_names=LABELS,
        digits=2, output_dict=True, zero_division=0
    )


    plot_f1_bars(rep, base)


    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plot_confusion_matrix(cm, base)


    acc_val = accuracy_score(y_true, y_pred)

    rows.append({
        "file": base,
        "acc": acc_val,
        "f1_neg": rep.get("Negative", {}).get("f1-score", 0.0),
        "f1_neu": rep.get("Neutral", {}).get("f1-score", 0.0),
        "f1_pos": rep.get("Positive", {}).get("f1-score", 0.0),
        "f1_macro": rep.get("macro avg", {}).get("f1-score", 0.0),
        "f1_weighted": rep.get("weighted avg", {}).get("f1-score", 0.0),
        "support": rep.get("accuracy", None)  
    })

summary = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=summary, x="f1_macro", y="file", palette="viridis")
plt.xlim(0,1)
plt.title("Comparação de F1-macro entre arquivos")
plt.xlabel("F1-macro")
plt.ylabel("Arquivo")
plt.savefig(os.path.join(OUTPUT_FOLDER, "_comparacao_f1_macro.png"), dpi=300, bbox_inches="tight")
plt.close()


summary.to_csv(os.path.join(OUTPUT_FOLDER, "_summary_with_f1.csv"), index=False, encoding="utf-8")
print("Gráficos e resumo salvos em:", OUTPUT_FOLDER)
