import os
import re
import json
import time
import requests
import pandas as pd
from typing import Optional, Tuple, List, Dict


DEFAULT_SYSTEM = "You are an expert in sentiment analysis."
USER_TEMPLATE = (
    "You are an expert in sentiment analysis. "
    "Classify the following YouTube music video comment as Positive, Negative, or Neutral, "
    "and justify your answer in a short sentence.\n\n"
    "Return ONLY a valid JSON object with EXACTLY two fields and NOTHING ELSE (no code fences):\n"
    "{{\"sentiment\":\"Positive|Negative|Neutral\",\"justification\":\"<short sentence>\"}}\n\n"
    "Comment:\n{comment}"
)


AUTO_COLUMNS = [
    "comment", "comentario", "comentários", "texto", "text",
    "conteudo", "content", "body", "review", "message"
]

def chamar_api(url: str, model: str, system_msg: str, user_msg: str, timeout: int = 60) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "stream": False
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "") or ""
    except requests.exceptions.RequestException as e:
        return f"__ERROR__: {e}"

def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE)
    return s.strip()

def _collapse_spaces(s: str) -> str:

    return re.sub(r"\s+", " ", (s or "").strip())

def parse_model_output(raw: str) -> Tuple[Optional[str], Optional[str]]:

    if not raw:
        return None, None

    if raw.startswith("__ERROR__"):
        return None, raw

    cleaned = _strip_code_fences(raw)

    try:
        obj = json.loads(cleaned)
        sent = obj.get("sentiment")
        just = obj.get("justification")
        if isinstance(sent, str) and isinstance(just, str):
            return _collapse_spaces(sent), _collapse_spaces(just)
    except Exception:
        pass

    m_sent = re.search(r'"sentiment"\s*:\s*"([^"]+)"', cleaned, flags=re.IGNORECASE)
    m_just = re.search(r'"justification"\s*:\s*"([^"]+)"', cleaned, flags=re.IGNORECASE)
    if m_sent and m_just:
        return _collapse_spaces(m_sent.group(1)), _collapse_spaces(m_just.group(1))

    lowered = cleaned.lower()
    guess = None
    if "positive" in lowered:
        guess = "Positive"
    elif "negative" in lowered:
        guess = "Negative"
    elif "neutral" in lowered:
        guess = "Neutral"

    if guess:
        return guess, _collapse_spaces(cleaned)

    return None, None

def find_comment_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def normalize_sentiment(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip().lower()
    if s in ("positive", "pos", "positivo"):
        return "Positive"
    if s in ("negative", "neg", "negativo"):
        return "Negative"
    if s in ("neutral", "neutro"):
        return "Neutral"
    return s.capitalize()

def analisar_csv(caminho_csv: str, url: str, model: str, output_dir: str):
    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(caminho_csv, encoding=enc, dtype=str)
            break
        except Exception:
            df = None
    if df is None:
        return f"Erro ao ler o CSV '{caminho_csv}'."

    col_comment = find_comment_column(df, AUTO_COLUMNS)
    if not col_comment:
        return f"Não encontrei coluna de comentário em '{caminho_csv}'. Colunas vistas: {list(df.columns)}"

    total = len(df)
    base = os.path.splitext(os.path.basename(caminho_csv))[0]
    ensure_dir(output_dir)

    registros: List[Dict[str, str]] = []
    print(f"Iniciando análise de sentimentos em {total} comentários de '{caminho_csv}'...\n")

    for idx, row in df.iterrows():
        comentario = (row.get(col_comment) or "").strip()
        if not comentario:
            registros.append({
                "arquivo": base,
                "linha_csv": idx + 2, 
                "comentario": "",
                "sentimento": "",
                "justificativa": "Linha vazia",
                "modelo": model,
                "timestamp": now_iso(),
                "resposta_bruta": ""
            })
            continue

        user_msg = USER_TEMPLATE.format(comment=comentario)
        raw = chamar_api(url, model, DEFAULT_SYSTEM, user_msg)
        sent, just = parse_model_output(raw)
        sent = normalize_sentiment(sent) or ""
        just = _collapse_spaces(just) or ""

        registros.append({
            "arquivo": base,
            "linha_csv": idx + 2, 
            "comentario": comentario.replace("\n", " ").strip(),
            "sentimento": sent,
            "justificativa": just,
            "modelo": model,
            "timestamp": now_iso(),
            "resposta_bruta": _collapse_spaces(raw),
        })

        if (idx + 1) % 10 == 0:
            print(f"Analisado {idx + 1}/{total} comentários...")

    print("\nAnálise concluída!\n")

    colunas_completas = [
        "arquivo", "linha_csv", "comentario",
        "sentimento", "justificativa",
        "modelo", "timestamp", "resposta_bruta"
    ]
    out_df = pd.DataFrame(registros, columns=colunas_completas)

    out_df_display = (
        out_df[["sentimento", "comentario", "justificativa"]]
        .rename(columns={"sentimento": "classificacao"})
    )

    analise_path = os.path.join(output_dir, f"{base}_analise.csv")
    out_df_display.to_csv(analise_path, index=False, encoding="utf-8-sig")
    print(f"Análise salva em '{analise_path}'")


    resumo = (
        out_df.assign(sentimento=lambda d: d["sentimento"].replace({"": "Indefinido"}))
              .groupby("sentimento", dropna=False)
              .size()
              .reset_index(name="quantidade")
              .sort_values("quantidade", ascending=False)
    )
    total_validos = resumo["quantidade"].sum()
    resumo["percentual"] = (resumo["quantidade"] / max(total_validos, 1) * 100).round(2)

    resumo_path = os.path.join(output_dir, f"{base}_resumo.csv")
    resumo.to_csv(resumo_path, index=False, encoding="utf-8-sig")
    print(f"Resumo salvo em '{resumo_path}'")

    return registros

def analisar_pasta(input_dir: str, url: str, model: str, output_dir: str):
    arquivos_csv = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    if not arquivos_csv:
        print("Nenhum arquivo CSV encontrado.")
        return

    for arquivo in arquivos_csv:
        caminho_csv = os.path.join(input_dir, arquivo)
        analisar_csv(caminho_csv, url, model, output_dir)

if __name__ == "__main__":

    url = "http://127.0.0.1:11434/api/chat"
    model = "gemma3:12b"

    input_dir = r"C:\Users\daniel\Desktop\LLM\comments"
    output_dir = r"C:\Users\daniel\Desktop\LLM\analise"

    os.makedirs(output_dir, exist_ok=True)
    analisar_pasta(input_dir, url, model, output_dir)
