import requests
import json
import pandas as pd
import time
import os

def chamar_api(url: str, model: str, system_msg: str, user_msg: str):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "Resposta não encontrada.")
    except requests.exceptions.RequestException as e:
        return f"Erro ao chamar a API: {str(e)}"

def analisar_csv(caminho_csv: str, url: str, model: str, output_dir: str):
    try:
        df = pd.read_csv(caminho_csv)
        resultados = []
        total_comentarios = len(df)
        print(f"Iniciando análise de sentimentos em {total_comentarios} comentários de '{caminho_csv}'...\n")
        for idx, comentario in enumerate(df['comment']):
            user_msg = f"Classify the following YouTube music video comment as Positive, Negative, or Neutral, and justify your answer in a short sentence.\n\n{comentario}"
            resposta_api = chamar_api(url, model, "You are an expert in sentiment analysis.", user_msg)
            resultados.append({
                "comentario": comentario,
                "resposta_api": resposta_api
            })
            if (idx + 1) % 10 == 0:
                print(f"Analisado {idx + 1}/{total_comentarios} comentários...")
        print("\nAnálise concluída!\n")
        nome_arquivo = os.path.splitext(os.path.basename(caminho_csv))[0]
        output_path = os.path.join(output_dir, f"{nome_arquivo}_analise.csv")
        resultados_df = pd.DataFrame(resultados)
        resultados_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Análise salva em '{output_path}'")
        return resultados
    except Exception as e:
        return f"Erro ao processar o arquivo CSV: {str(e)}"

def analisar_pasta(input_dir: str, url: str, model: str, output_dir: str):
    arquivos_csv = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not arquivos_csv:
        print("Nenhum arquivo CSV encontrado.")
        return
    for arquivo in arquivos_csv:
        caminho_csv = os.path.join(input_dir, arquivo)
        analisar_csv(caminho_csv, url, model, output_dir)

if __name__ == "__main__":
    url = "http://127.0.0.1:11434/api/chat"
    model = "llama3:8b"
    input_dir = 'C:\\Users\\daniel\\Desktop\\LLM\\comments'
    output_dir = 'C:\\Users\\daniel\\Desktop\\LLM\\analise'
    os.makedirs(output_dir, exist_ok=True)
    analisar_pasta(input_dir, url, model, output_dir)
