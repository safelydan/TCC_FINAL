# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURAÇÕES
# =========================
INPUT_FOLDER  = r"C:\Users\daniel\Desktop\LLM_MANUAL\LLM_FEW_SHOT\ANALISE_MANUAL_LLM_EW_SHOTS"
OUTPUT_FOLDER = r"C:\Users\daniel\Desktop\LLM_MANUAL\LLM_FEW_SHOT\GRAFICOS_MANUAL"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

summary_csv = os.path.join(OUTPUT_FOLDER, "resumo_agregado.csv")

# =========================
# FUNÇÕES
# =========================
def should_skip_file(path: str) -> bool:
    """
    Pule arquivos de RESUMO/AGREGADOS pelo nome.
    """
    name = os.path.basename(path).lower()
    # pula qualquer coisa com 'resumo' no nome ou o próprio arquivo agregado que vamos gerar
    return (
        "resumo" in name
        or "aggregate" in name
        or "aggregado" in name
        or name == os.path.basename(summary_csv).lower()
    )

def normalize_label(value):
    """
    Normaliza valores para {Positive, Neutral, Negative}.
    Aceita strings (case-insensitive) e NÚMEROS SOMENTE em {-1, 0, 1}.
    'Mixed' vira 'Neutral'.
    Outros números (ex.: 3, 100, 0.25) são ignorados (retorna None).
    """
    if pd.isna(value):
        return None

    # tenta como número, aceitando apenas -1, 0, 1
    try:
        v = float(str(value).strip().replace(",", "."))
        if v in (-1.0, 0.0, 1.0):
            if v > 0:  return "Positive"
            if v < 0:  return "Negative"
            return "Neutral"
        return None
    except Exception:
        pass

    s = str(value).strip().lower()
    mapping = {
        "positive": "Positive", "pos": "Positive", "1": "Positive",
        "negative": "Negative", "neg": "Negative", "-1": "Negative",
        "neutral": "Neutral",  "neu": "Neutral",  "0": "Neutral",
        "mixed": "Neutral",
    }
    return mapping.get(s, None)

def count_sentiments(series: pd.Series) -> dict:
    """
    Conta Positive/Neutral/Negative na série, após normalização.
    """
    normalized = series.map(normalize_label).dropna()
    counts = normalized.value_counts().to_dict()
    for k in ["Positive", "Neutral", "Negative"]:
        counts.setdefault(k, 0)
    return counts

def detect_manual_column(df: pd.DataFrame) -> str | None:
    """
    Retorna SOMENTE coluna de classificação MANUAL (forma estrita).
    Aceita apenas nomes exatos comuns: 'sentimento_manual' ou 'classificacao_manual'.
    Não aceita qualquer coluna que contenha 'manual' para evitar pegar percentuais de resumo.
    """
    lower_cols = {c.lower(): c for c in df.columns}
    for exact in ["sentimento_manual", "classificacao_manual"]:
        if exact in lower_cols:
            return lower_cols[exact]
    return None

# =========================
# PIPELINE
# =========================
# lista todos os CSVs
all_csvs = sorted(glob.glob(os.path.join(INPUT_FOLDER, "**", "*.csv"), recursive=True))
# pula os 'resumo'
csv_files = [p for p in all_csvs if not should_skip_file(p)]

if not csv_files:
    raise SystemExit(f"Nenhum CSV (não-resumo) encontrado em: {INPUT_FOLDER}")

per_file_rows = []
for csv_path in csv_files:
    # leitura robusta
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path, encoding="latin-1", sep=None, engine="python")

    col = detect_manual_column(df)
    if col is None:
        print(f"Aviso: coluna manual ('sentimento_manual' ou 'classificacao_manual') não encontrada em {csv_path}. Pulando.")
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
    raise SystemExit("Nenhum arquivo válido processado (sem coluna manual válida).")

# Agregado e CSV
agg_df = pd.DataFrame(per_file_rows).sort_values("arquivo").reset_index(drop=True)
agg_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

# Soma global e gráficos (sem cores específicas)
sum_pos = int(agg_df["Positive"].sum())
sum_neu = int(agg_df["Neutral"].sum())
sum_neg = int(agg_df["Negative"].sum())

labels = ["Positive", "Neutral", "Negative"]
counts = [sum_pos, sum_neu, sum_neg]

# Bar chart (não definir cores)
plt.figure(figsize=(8, 5))
plt.bar(labels, counts)
plt.xlabel("Sentiment")
plt.ylabel("Comment Count")
plt.title("Sentiment Analysis (ALL FILES) - Bar (MANUAL)")
bar_path = os.path.join(OUTPUT_FOLDER, "sentiment_bar_all.png")
plt.tight_layout()
plt.savefig(bar_path, dpi=300)
plt.close()

# Pie chart (não definir cores)
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct="%1.1f%%")
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
