import os
import re
import string
import pandas as pd
from time import sleep
from urllib.parse import quote
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import requests
from difflib import SequenceMatcher

# =========================
# Limpeza e Normalização
# =========================

def clean_song_title(title: str) -> str:
    """Remove termos como 'remastered', 'live', 'hd' do título para melhorar a busca da letra."""
    return re.sub(r'\(.*?\)|\[.*?\]|Remastered|Live|HD|Official', '', title, flags=re.IGNORECASE).strip()

def normalize_filename(name: str) -> str:
    """Normaliza o nome do arquivo para evitar problemas no sistema de arquivos."""
    valid_chars = f"-_.'() {string.ascii_letters}{string.digits}"
    return ''.join(c for c in name if c in valid_chars).lower()

_punct_re = re.compile(r'[^\w\s]', re.UNICODE)
_ws_re = re.compile(r'\s+')

def normalize_comment(text: str) -> str:
    """
    Normaliza o comentário para comparação:
    - minúsculas
    - remove pontuação
    - colapsa espaços
    """
    t = text.lower().strip()
    t = _punct_re.sub(' ', t)
    t = _ws_re.sub(' ', t)
    return t.strip()

# =========================
# Cache simples para letras
# =========================

lyrics_cache = {}

def get_lyrics(song_title: str, artist: str = "") -> list[str]:
    """Obtém e normaliza a letra da música (linha a linha) com cache."""
    song_title = clean_song_title(song_title)
    key = f"{artist}_{song_title}".lower()
    if key in lyrics_cache:
        return lyrics_cache[key]

    # Codifica para URL
    url = f"https://api.lyrics.ovh/v1/{quote(artist)}/{quote(song_title)}"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            lyrics = [
                normalize_comment(line)
                for line in response.json().get("lyrics", "").split('\n')
                if line.strip()
            ]
            lyrics_cache[key] = lyrics
            return lyrics
    except requests.RequestException:
        pass
    return []

def is_similar_to_lyrics(comment: str, lyrics_norm_lines: list[str], threshold: float = 0.8) -> bool:
    """Verifica se o comentário é similar a alguma linha da letra (evita capturar o trecho da música)."""
    comment_norm = normalize_comment(comment)
    for line in lyrics_norm_lines:
        if not line:
            continue
        if SequenceMatcher(None, comment_norm, line).ratio() >= threshold:
            return True
    return False

def is_verse(comment: str) -> bool:
    """Verifica se o comentário parece uma estrofe (mais de uma linha)."""
    # Considera quebras de linha explícitas como indício de estrofe/trecho copiado
    return len([ln for ln in comment.splitlines() if ln.strip()]) > 1

def is_near_duplicate(comment: str, accepted_norms: list[str], threshold: float = 0.92) -> bool:
    """
    Compara o comentário normalizado com os já aceitos.
    Usa SequenceMatcher para detectar "quase iguais" (ex.: diferença só de pontuação/espaço).
    """
    c_norm = normalize_comment(comment)
    for prev in accepted_norms:
        if SequenceMatcher(None, c_norm, prev).ratio() >= threshold:
            return True
    return False

# =========================
# YouTube helpers
# =========================

def get_video_ids_and_titles(api_key: str, playlist_id: str) -> list[tuple[str, str]]:
    """Obtém ids e títulos dos vídeos da playlist especificada."""
    youtube = build('youtube', 'v3', developerKey=api_key)
    videos = []
    request = youtube.playlistItems().list(part="snippet", playlistId=playlist_id, maxResults=50)
    while request:
        response = request.execute()
        for item in response.get('items', []):
            video_id = item['snippet']['resourceId']['videoId']
            title = item['snippet']['title']
            videos.append((video_id, title))
        request = youtube.playlistItems().list_next(request, response)
    return videos

def get_comments(
    api_key: str,
    video_id: str,
    title: str,
    keywords: list[str],
    max_comments: int = 300,
    output_dir: str = "../data/comments",
    lyric_similarity_threshold: float = 0.80,
    near_dup_threshold: float = 0.92
) -> None:
    """
    Extrai comentários principais, com paginação explícita.
    Evita estrofes, trechos de letra, duplicados e quase‑duplicados.
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"{normalize_filename(title)} comments.csv")
    if os.path.exists(filename):
        print(f"arquivo '{filename}' já existe. pulando {title}...")
        return

    # DF final
    df = pd.DataFrame(columns=["comment", "user_name", "date", "likes"])

    # Deduplicação
    seen_exact = set()
    accepted_norms: list[str] = []

    # Letras normalizadas
    lyrics_norm = get_lyrics(title)

    # Palavras‑chave
    keyword_pattern = re.compile('|'.join(re.escape(w) for w in keywords), re.IGNORECASE)

    comment_count = 0
    page_token = None
    page_idx = 0

    while True:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=100,          # máximo permitido
                order="time",            # varre cronologicamente para pegar mais páginas
                pageToken=page_token     # paginação explícita
            )
            response = request.execute()
            page_idx += 1

            batch_comments, batch_dates, batch_user_names, batch_like_counts = [], [], [], []

            items = response.get('items', [])
            if not items:
                # Sem itens nesta página -> fim
                break

            for item in items:
                snippet = item['snippet']['topLevelComment']['snippet']
                comment = snippet.get('textDisplay', '') or ''
                user = snippet.get('authorDisplayName', '') or ''
                date = snippet.get('publishedAt', '') or ''
                likes = int(snippet.get('likeCount', 0))

                # Filtros
                if not comment.strip():
                    continue
                if is_verse(comment):
                    continue
                if not keyword_pattern.search(comment):
                    continue
                if lyrics_norm and is_similar_to_lyrics(comment, lyrics_norm, threshold=lyric_similarity_threshold):
                    continue

                # Duplicados
                if comment in seen_exact:
                    continue
                if is_near_duplicate(comment, accepted_norms, threshold=near_dup_threshold):
                    continue

                # Aceita
                seen_exact.add(comment)
                accepted_norms.append(normalize_comment(comment))
                batch_comments.append(comment)
                batch_user_names.append(user)
                batch_dates.append(date)
                batch_like_counts.append(likes)

                comment_count += 1
                if comment_count >= max_comments:
                    break

            # Atualiza DF e salva incrementalmente
            if batch_comments:
                df2 = pd.DataFrame({
                    "comment": batch_comments,
                    "user_name": batch_user_names,
                    "date": batch_dates,
                    "likes": batch_like_counts
                })
                df = pd.concat([df, df2], ignore_index=True)
                df = df.drop_duplicates(subset=["comment"], keep="first")
                df = df.sort_values(by="likes", ascending=False)
                df.head(max_comments).to_csv(filename, index=False, encoding='utf-8')

            if comment_count >= max_comments:
                # Atingiu o limite desejado
                break

            # Próxima página
            page_token = response.get('nextPageToken')
            if not page_token:
                # Não há mais páginas
                break

            sleep(1.5)  # respeita cota

        except HttpError as e:
            if "commentsDisabled" in str(e):
                print(f"comentários desativados para '{title}' ({video_id}). pulando...")
            else:
                print(f"erro ao processar '{title}' ({video_id}): {e}")
            break
        except Exception as e:
            print(f"erro inesperado em '{title}': {e}")
            break

# =========================
# Main
# =========================

def main():
    api_key = 'AIzaSyD_0nNX7jPiWLiTxEZ17_22oHkyMbB_ny8'
    playlist_id = 'PL5vXKG03DXKFeOZrrCsZJccJUeBbkBXM2'

    keywords = [
        # aspectos musicais gerais
        "melody", "harmony", "rhythm", "tempo", "beat", "chords", "notes",
        "tone", "timbre", "composition", "progression", "pitch", "acoustic",

        # elementos vocais e letra
        "lyrics", "verse", "chorus", "vocal",

        # produção musical e mixagem
        "mixing", "mastering", "reverb", "distortion", "effects", "synth", "autotune",

        # emoções e impacto sonoro
        "emotion", "soulful", "powerful", "deep",

        # impacto e conexão emocional
        "catchy", "vibe", "nostalgic", "chills",

        # narrativa e significado
        "story", "message",

        # críticas negativas
        "boring", "repetitive", "generic", "forgettable", "lifeless", "annoying", "monotonous", "weak",

        # termos fortes
        "terrible", "worst", "awful"
    ]

    videos = get_video_ids_and_titles(api_key, playlist_id)
    print(f"total de vídeos encontrados: {len(videos)}\n")

    for video_id, title in videos:
        print(f"iniciando processamento: {title} - {video_id}")
        get_comments(api_key, video_id, title, keywords)
        print(f"processamento finalizado para: {title}\n")

if __name__ == "__main__":
    main()
