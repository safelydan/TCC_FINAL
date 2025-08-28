import os
import re
import json
import time
import requests
import pandas as pd
from typing import Optional, Tuple, List, Dict

# =========================
# Configuração do Prompt
# =========================

DEFAULT_SYSTEM = "You are an expert in sentiment analysis."

PROMPT_HEADER = (
    "You are an expert in sentiment analysis. "
    "Classify the following YouTube music video comment as Positive, Negative, Neutral, or Mixed, "
    "and justify your answer in a short sentence.\n\n"
    "Return ONLY a valid JSON object with EXACTLY two fields and NOTHING ELSE (no code fences):\n"
    "{\"sentiment\":\"Positive|Negative|Neutral|Mixed\",\"justification\":\"<short sentence>\"}\n\n"
    "Follow the output style shown in the examples below.\n"
)

# Exemplos few-shot (NEGATIVE, POSITIVE, NEUTRAL, MIXED)
FEW_SHOT = [
    # === NEGATIVE ===
    (
        "this won a grammy?? well its givin DEI vibes 😂😂😂😂😂😂😂😂😂😂😂😂😂😂😂😂😂😂😂😂",
        '{"sentiment":"Negative","justification":"Sarcastic remark ridiculing the Grammy win."}'
    ),
    (
        "The newest generation has ruined music! Just terrible, horrible singing, terrible melody…. You are all lost in ur selfie addiction and TikTok waste of time…",
        '{"sentiment":"Negative","justification":"Strongly criticizes modern music and culture in harsh terms."}'
    ),
    (
        "Modern day lyrics is so immoral and no class. Make the genitalia water? Really?",
        '{"sentiment":"Negative","justification":"Condemns lyrics as immoral and classless."}'
    ),
    (
        "She did it again. Olivia somehow managed to be more generic than Taylor Swift. She carefully handpicked another one of the most generic situations in life and is pretending she wrote and filmed a masterpiece. The delusion is beyond. And the people who likes her music are the most basic I ever witnessed in my lifetime. That's impressive. What a load of shit.",
        '{"sentiment":"Negative","justification":"Dismisses Olivia\'s song as generic and insulting to listeners."}'
    ),
    (
        "This morbid song really sucks , one of the top 5 worst songs of all time",
        '{"sentiment":"Negative","justification":"Calls the song terrible and among the worst ever."}'
    ),

    # === POSITIVE ===
    (
        "I'm a 27 year old straight black man and I love this song. I also like grins, you, and what  I like. What I like is special because I first heard it the summer before I went to college in 2014, I remember being so enthusiastic and happy about leaving my hometown and never having to go back to high school which I hated , yes I know it came out in 2013 but 2014 was when I first heard it . Everytime I hear it the memories and emotions from that summer come back",
        '{"sentiment":"Positive","justification":"Expresses deep love for the song and associates it with joyful memories."}'
    ),
    (
        "I LOVE feel good music! The kind that makes you feel like you’re in the 1960’s floating on a cloud sipping lemonade in the summer with a nice breeze!!! “Say So” by Doja Cat gives me same vibe! Love this!",
        '{"sentiment":"Positive","justification":"Describes euphoric feelings and strong enjoyment of the music."}'
    ),
    (
        "This is thrilling. Everything we could have wanted from her new era - it feels like all previous eras of Gaga are present on this record - we hear the virtuoso pianist, the rock star, the pop star, the jazz vocalist taking risks, the dark synth queen … and a totally unique artistic voice - one that can look at the whole of humanity good and bad and deliver something of beauty - with depth and emotional range and yet the same time - also still, to put simply - a catchy hook laden pop song! No one can touch her! Thrilling!",
        '{"sentiment":"Positive","justification":"Extensive praise for Gaga\'s artistry and uniqueness."}'
    ),
    (
        "A true masterpiece of art and creativity. The vocals are impeccable. The passion of music is life. Thank you Gaga for your time and works.",
        '{"sentiment":"Positive","justification":"Labels it a masterpiece with impeccable vocals and passion."}'
    ),
    (
        "This song is a lot more harmonically sophisticated than it seems at first. The modal shifts and dissonant chords really get under your skin... in the best possible way.",
        '{"sentiment":"Positive","justification":"Highlights harmonic sophistication and praises the effect."}'
    ),

    # === NEUTRAL ===
    (
        "First verse sounds so much like Rihanna It's just ok. I wish the lyrics were a little better.",
        '{"sentiment":"Neutral","justification":"Mixed reaction without strong polarity."}'
    ),
    (
        "Well...not that great tbh (sorry for sharing opinion). Melody is to broken (chorus is ok), voice as always is good, lyrics a bit... juvenile (and give me cure vibe) - just not what I was expecting tbh",
        '{"sentiment":"Neutral","justification":"Balanced remarks with positives and negatives."}'
    ),
    (
        "Chorus is catchy but overall sounds dated. Don\'t think this\'ll chart well. Hope I\'m wrong - excited for the album.",
        '{"sentiment":"Neutral","justification":"Notes pros and cons and expresses uncertainty."}'
    ),
    (
        "I love Sabrina so much but I really think she doesn’t deserve Best Pop Vocal Album Sabrina i love you so much and you’re slay album but there is no way your vocals win Ariana vocals,@adirdayan123,2025-07-10T14:00:48Z,2",
        '{"sentiment":"Neutral","justification":"Affection plus criticism; overall mixed stance without strong polarity."}'
    ),

    # === MIXED (5 exemplos) ===
    (
        "I love Sabrina so much but I really think she doesn’t deserve Best Pop Vocal Album.",
        '{"sentiment":"Mixed","justification":"Shows affection for the artist but criticizes the award recognition."}'
    ),
    (
        "The beat is amazing, but the lyrics feel very shallow and repetitive.",
        '{"sentiment":"Mixed","justification":"Praises the beat but criticizes the lyrics."}'
    ),
    (
        "Her voice is beautiful, but the production ruins the vibe of the song.",
        '{"sentiment":"Mixed","justification":"Compliments the vocals but disapproves of the production."}'
    ),
    (
        "This song reminds me of my childhood which is nice, but honestly it sounds outdated now.",
        '{"sentiment":"Mixed","justification":"Positive nostalgia yet calls the sound outdated."}'
    ),
    (
        "Great chorus and energy, but the verses drag and make it hard to replay.",
        '{"sentiment":"Mixed","justification":"Strong chorus contrasted with weak verses."}'
    ),
]

USE_FEW_SHOT = True

def build_user_prompt(comment: str) -> str:
    if not USE_FEW_SHOT or not FEW_SHOT:
        return (
            "You are an expert in sentiment analysis. "
            "Classify the following YouTube music video comment as Positive, Negative, Neutral, or Mixed, "
            "and justify your answer in a short sentence.\n\n"
            "Return ONLY a valid JSON object with EXACTLY two fields and NOTHING ELSE (no code fences):\n"
            "{\"sentiment\":\"Positive|Negative|Neutral|Mixed\",\"justification\":\"<short sentence>\"}\n\n"
            f"Comment:\n{comment}"
        )

    examples = []
    for ex_in, ex_out in FEW_SHOT:
        examples.append(
            "Example\n"
            "Comment:\n" + ex_in + "\n"
            "Assistant:\n" + ex_out + "\n"
        )
    examples_block = "\n".join(examples).strip()

    final_task = (
        "Now classify the NEW comment below. "
        "Return ONLY the JSON object, with no extra text.\n\n"
        f"Comment:\n{comment}"
    )

    return PROMPT_HEADER + "\n" + examples_block + "\n\n" + final_task

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
    if "mixed" in lowered:
        guess = "Mixed"
    elif "positive" in lowered:
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
    if s in ("mixed", "misto", "misturado"):
        return "Mixed"
    return s.capitalize()

def _tentar_reaproveitar_sentimento_manual(analise_path_novo: str, df_novo: pd.DataFrame) -> pd.Series:

    if not os.path.exists(analise_path_novo):
        return pd.Series([""] * len(df_novo), index=df_novo.index, dtype=object)

    try:
        antigo = pd.read_csv(analise_path_novo, dtype=str, encoding="utf-8-sig")
    except Exception:
        return pd.Series([""] * len(df_novo), index=df_novo.index, dtype=object)

    if "comentario" not in antigo.columns or "sentimento_manual" not in antigo.columns:
        return pd.Series([""] * len(df_novo), index=df_novo.index, dtype=object)

    mapa = antigo.set_index("comentario")["sentimento_manual"].to_dict()
    return df_novo["comentario"].map(mapa).fillna("")

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

        user_msg = build_user_prompt(comentario)

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

    sent_manual_series = _tentar_reaproveitar_sentimento_manual(analise_path, out_df_display)
    out_df_display.insert(1, "sentimento_manual", sent_manual_series.astype(str))

    out_df_display.to_csv(analise_path, index=False, encoding="utf-8-sig")
    print(f"Análise salva em '{analise_path}'")

    resumo = (
        out_df.assign(sentimento=lambda d: d["sentimento"].replace({"": "Indefinido"}))
              .groupby("sentimento", dropna=False)
              .size()
              .reset_index(name="quantidade")
              .sort_values("quantidade", ascending=False)
              .reset_index(drop=True)
    )
    total_validos = resumo["quantidade"].sum()
    resumo["percentual"] = (resumo["quantidade"] / max(total_validos, 1) * 100).round(2)

    resumo["sentimento_manual"] = ""

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
    model = "gemma3:4b"

    input_dir = r"C:\Users\daniel\Desktop\LLM_FINAL\data\comentarios_coletados"
    output_dir = r"C:\Users\daniel\Desktop\LLM\analise"

    os.makedirs(output_dir, exist_ok=True)
    analisar_pasta(input_dir, url, model, output_dir)
