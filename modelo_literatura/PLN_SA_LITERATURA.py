# -*- coding: utf-8 -*-
"""
batch_csv_sentiment.py

Adaptação do código para processar TODOS os CSVs de uma pasta.
- Lê cada CSV que contenha uma coluna 'comment' (ajuste o nome se necessário).
- Calcula o compound score (VADER), cria a coluna 'sentiment' (Positive/Neutral/Negative).
- Salva uma cópia anotada por arquivo e um resumo geral (CSV) com contagens e percentuais.
- Gera gráficos de barras e pizza do agregado (todos os arquivos).

Requisitos:
    pip install pandas matplotlib vaderSentiment
"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ======================== CONFIGURAÇÕES ========================
INPUT_FOLDER = r"C:\Users\daniel\Desktop\LLM\comments"   
OUTPUT_FOLDER = r"C:\Users\daniel\Desktop\LLM_MANUAL\modelo_literatura\resultados"   # Pasta de saída para arquivos e gráficos
CSV_GLOB = "*.csv"                    # Padrão de arquivos (ex.: '*.csv')
COMMENT_COLUMN = "comment"            # Nome da coluna com os comentários
ENCODING = "utf-8-sig"                # 'utf-8', 'utf-8-sig', etc.
POS_THRESH = 0.05                     # Threshold positivo
NEG_THRESH = -0.05                    # Threshold negativo
# ===============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
analyzer = SentimentIntensityAnalyzer()

def classify_score(compound: float) -> str:
    if compound > POS_THRESH:
        return "Positive"
    elif compound < NEG_THRESH:
        return "Negative"
    return "Neutral"

def process_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=ENCODING)
    if COMMENT_COLUMN not in df.columns:
        raise ValueError(f"O arquivo '{os.path.basename(path)}' não possui a coluna '{COMMENT_COLUMN}'.")
    # Calcula polaridade por linha
    scores = df[COMMENT_COLUMN].astype(str).apply(lambda t: analyzer.polarity_scores(t)["compound"])
    df["compound"] = scores
    df["sentiment"] = df["compound"].apply(classify_score)
    return df

def summarize_sentiments(df: pd.DataFrame, file_label: str) -> dict:
    total = len(df)
    pos = (df["sentiment"] == "Positive").sum()
    neu = (df["sentiment"] == "Neutral").sum()
    neg = (df["sentiment"] == "Negative").sum()
    avg = df["compound"].mean() if total else 0.0
    if avg > POS_THRESH:
        overall = "Positive"
    elif avg < NEG_THRESH:
        overall = "Negative"
    else:
        overall = "Neutral"
    return {
        "file": file_label,
        "total": total,
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "pct_positive": (pos/total*100) if total else 0.0,
        "pct_neutral": (neu/total*100) if total else 0.0,
        "pct_negative": (neg/total*100) if total else 0.0,
        "avg_compound": avg,
        "overall_sentiment": overall,
    }

def main():
    files = sorted(glob.glob(os.path.join(INPUT_FOLDER, CSV_GLOB)))
    if not files:
        print(f"Nenhum CSV encontrado em: {INPUT_FOLDER}")
        return

    all_rows = []
    per_file_summaries = []
    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"Processando: {fname}")
        try:
            df = process_file(fpath)
        except Exception as e:
            print(f"  [ERRO] {fname}: {e}")
            continue

        # Salva arquivo anotado
        out_csv = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(fname)[0]}_scored.csv")
        df.to_csv(out_csv, index=False, encoding=ENCODING)

        # Most positive / most negative
        if not df.empty:
            idx_pos = df["compound"].idxmax()
            idx_neg = df["compound"].idxmin()
            print(f"  Comentário mais positivo (score={df.loc[idx_pos,'compound']:.4f}): {df.loc[idx_pos, COMMENT_COLUMN][:200]}")
            print(f"  Comentário mais negativo (score={df.loc[idx_neg,'compound']:.4f}): {df.loc[idx_neg, COMMENT_COLUMN][:200]}")

        # Resumo por arquivo
        summary = summarize_sentiments(df, fname)
        per_file_summaries.append(summary)

        # Acumula para agregado global
        all_rows.append(df[["compound", "sentiment"]])

    # Agregado geral (todos os arquivos)
    if all_rows:
        agg = pd.concat(all_rows, ignore_index=True)
        agg_summary = summarize_sentiments(agg, "ALL_FILES")
        per_file_summaries.append(agg_summary)

        # Salva resumo geral
        summary_df = pd.DataFrame(per_file_summaries)
        summary_csv = os.path.join(OUTPUT_FOLDER, "summary_sentiment.csv")
        summary_df.to_csv(summary_csv, index=False, encoding=ENCODING)

        # Gráficos (agregado)
        labels = ["Positive", "Negative", "Neutral"]
        counts = [
            (agg["sentiment"] == "Positive").sum(),
            (agg["sentiment"] == "Negative").sum(),
            (agg["sentiment"] == "Neutral").sum(),
        ]

        # Bar chart
        plt.figure(figsize=(8, 5))
        plt.bar(labels, counts)  # Não especificar cores
        plt.xlabel("Sentiment")
        plt.ylabel("Comment Count")
        plt.title("Sentiment Analysis (ALL FILES) - Bar")
        bar_path = os.path.join(OUTPUT_FOLDER, "sentiment_bar_all.png")
        plt.tight_layout()
        plt.savefig(bar_path, dpi=300)
        plt.close()

        # Pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(counts, labels=labels, autopct="%1.1f%%")
        plt.title("Sentiment Analysis (ALL FILES) - Pie")
        pie_path = os.path.join(OUTPUT_FOLDER, "sentiment_pie_all.png")
        plt.tight_layout()
        plt.savefig(pie_path, dpi=300)
        plt.close()

        print("\nResumo agregado (todos os arquivos):")
        print(agg_summary)
        print(f"\nArquivos gerados em: {OUTPUT_FOLDER}")
        print(f"- Resumo CSV: {summary_csv}")
        print(f"- Gráfico de barras: {bar_path}")
        print(f"- Gráfico de pizza: {pie_path}")
    else:
        print("Nenhum dado processado.")

if __name__ == "__main__":
    main()
