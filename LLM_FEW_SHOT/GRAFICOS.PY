# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURAÇÕES
# =========================
# Ajuste para a sua pasta com CSVs:
INPUT_FOLDER  = r"C:\Users\daniel\Desktop\LLM_MANUAL\LLM_FEW_SHOT\ANALISE_MANUAL_LLM_EW_SHOTS"
OUTPUT_FOLDER = r"C:\Users\daniel\Desktop\LLM_MANUAL\LLM_FEW_SHOT\GRAFICOS"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

summary_csv = os.path.join(OUTPUT_FOLDER, "resumo_agregado.csv")

# =========================
# FUNÇÕES
# =========================
def normalize_label(value):
    """Normaliza valores para {Positive, Neutral, Negative}.
       Aceita strings (case-insensitive) e numéricos (1,0,-1).
       'Mixed' vira 'Neutral'.
    """
    if pd.isna(value):
        return None

    # tenta numérico
    try:
        v = float(str(value).strip().replace(",", "."))
        if v > 0:  return "Positive"
        if v < 0:  return "Negative"
        return "Neutral"
    except Exception:
        pass

    s = str(value).strip().lower()
    mapping = {
        "positive": "Positive", "pos": "Positive", "1": "Positive",
        "negative": "Negative", "neg": "Negative", "-1": "Negative",
        "neutral": "Neutral", "neu": "Neutral", "0": "Neutral",
        "mixed": "Neutral",
    }
    return mapping.get(s, None)

def count_sentiments(series: pd.Series) -> dict:
    """Conta Positive/Neutral/Negative na série."""
    normalized = series.map(normalize_label).dropna()
    counts = normalized.value_counts().to_dict()
    for k in ["Positive", "Neutral", "Negative"]:
        counts.setdefault(k, 0)
    return counts

def detect_classificacao_column(df: pd.DataFrame) -> str | None:
    """Retorna somente a coluna 'classificacao'.
       Fallback: primeira coluna que contenha 'classific' no nome.
       Nunca retorna colunas com 'sentim' ou 'manual'.
    """
    lower_cols = {c.lower(): c for c in df.columns}
    if "classificacao" in lower_cols:
        return lower_cols["classificacao"]

    # fallback por substring
    for c in df.columns:
        cl = str(c).lower()
        if ("classific" in cl) and ("sentim" not in cl) and ("manual" not in cl):
            return c
    return None

# =========================
# PIPELINE
# =========================
csv_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "**", "*.csv"), recursive=True))
if not csv_files:
    raise SystemExit(f"Nenhum CSV encontrado em: {INPUT_FOLDER}")

per_file_rows = []
for csv_path in csv_files:
    # leitura robusta
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path, encoding="latin-1", sep=None, engine="python")

    col = detect_classificacao_column(df)
    if col is None:
        print(f"Aviso: coluna 'classificacao' não encontrada em {csv_path}. Pulando.")
        continue

    counts = count_sentiments(df[col])
    total = sum(counts.values())
    pos, neu, neg = counts["Positive"], counts["Neutral"], counts["Negative"]

    per_file_rows.append({
        "arquivo": os.path.basename(csv_path),
        "Positive": pos,
        "Neutral": neu,
        "Negative": neg,
        "Total": total,
        "Positive_%": (pos / total * 100) if total else 0.0,
        "Neutral_%": (neu / total * 100) if total else 0.0,
        "Negative_%": (neg / total * 100) if total else 0.0,
    })

if not per_file_rows:
    raise SystemExit("Nenhum arquivo válido processado (sem coluna 'classificacao').")

# DataFrame agregado e CSV
agg_df = pd.DataFrame(per_file_rows).sort_values("arquivo").reset_index(drop=True)
agg_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

# Soma global e gráficos (sem especificar cores)
sum_pos = int(agg_df["Positive"].sum())
sum_neu = int(agg_df["Neutral"].sum())
sum_neg = int(agg_df["Negative"].sum())

labels = ["Positive", "Neutral", "Negative"]
counts = [sum_pos, sum_neu, sum_neg]

# Bar chart (cores melhores)
plt.figure(figsize=(8, 5))
plt.bar(labels, counts, color=['#2ecc71', '#95a5a6', '#e74c3c'])  # verde, cinza, vermelho
plt.xlabel("Sentiment")
plt.ylabel("Comment Count")
plt.title("Sentiment Analysis (ALL FILES) - Bar (MANUAL)")
bar_path = os.path.join(OUTPUT_FOLDER, "sentiment_bar_all.png")
plt.tight_layout()
plt.savefig(bar_path, dpi=300)
plt.close()

# Pie chart (cores melhores)
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct="%1.1f%%",
        colors=['#2ecc71', '#95a5a6', '#e74c3c'])
plt.title("Sentiment Analysis (ALL FILES) - Pie (MANUAL)")
pie_path = os.path.join(OUTPUT_FOLDER, "sentiment_pie_all.png")
plt.tight_layout()
plt.savefig(pie_path, dpi=300)
plt.close()

# Resumo agregado no console
total_all = sum(counts) if sum(counts) else 1
agg_summary = pd.DataFrame({
    "Sentiment": labels,
    "Count": counts,
    "Percent": [c / total_all * 100 for c in counts]
})

print("\nResumo agregado (todos os arquivos):")
print(agg_summary)
print(f"\nArquivos gerados em: {OUTPUT_FOLDER}")
print(f"- Resumo CSV: {summary_csv}")
print(f"- Gráfico de barras: {bar_path}")
print(f"- Gráfico de pizza: {pie_path}")
