
# Análise de Sentimentos para Comentários do YouTube

Este script realiza análise de sentimentos em comentários extraídos de vídeos musicais do YouTube. Ele utiliza um modelo de linguagem (LLM) local através do **Ollama** para classificar os comentários como **Positivos**, **Negativos** ou **Neutros**, e também fornece uma justificativa para cada classificação.

## Pré-requisitos

Antes de rodar o código, você precisa ter as seguintes ferramentas e bibliotecas instaladas:

### Ferramentas
- [Ollama](https://ollama.com/download): Instale o Ollama em seu computador para rodar os modelos de linguagem localmente.
- [Python 3.6+](https://www.python.org/downloads/): Certifique-se de ter o Python instalado em seu sistema.

### Bibliotecas Python
As bibliotecas necessárias são listadas no arquivo `requirements.txt`.

Você pode instalar as dependências com o comando:

```bash
pip install -r requirements.txt
```

## Estrutura do Projeto

- `analisar_csvs_ollama.py`: Script principal que realiza a análise de sentimentos dos comentários.
- `requirements.txt`: Lista de dependências necessárias.
- `analise_resultados.csv`: Arquivo gerado com os resultados da análise (será salvo na mesma pasta onde o script é executado).

## Como Usar

### 1. Configuração do Ollama

Certifique-se de que o Ollama está instalado e rodando corretamente na sua máquina. Para isso, abra o terminal e execute o seguinte comando para baixar o modelo que será usado para análise (por exemplo, `llama3:8b`):

```bash
ollama pull llama3:8b
```

Verifique se o servidor do Ollama está ativo, rodando na porta `11434`:

```bash
curl http://127.0.0.1:11434/api/tags
```

### 2. Executando o Código

- Substitua o caminho do arquivo CSV na variável `caminho_csv` no script.
- Execute o script utilizando o comando:

```bash
python analisar_csvs_ollama.py
```

O script irá analisar os comentários presentes no arquivo CSV e salvar os resultados em um novo arquivo `analise_resultados.csv`.

### 3. Resultados

O arquivo `analise_resultados.csv` será gerado com as seguintes colunas:
- `comentario`: O comentário original do YouTube.
- `resposta_api`: A resposta da API, contendo a classificação do comentário (Positivo, Negativo ou Neutro) e uma justificativa para essa classificação.

### Exemplo de Saída

O arquivo `analise_resultados.csv` terá o seguinte formato:

| comentario                          | resposta_api                                                      |
|-------------------------------------|-------------------------------------------------------------------|
| "Amo essa música, muito boa!"       | {"sentiment": "Positive", "justification": "Muito elogiada pela emoção."} |
| "Não gostei dessa música, horrível!"| {"sentiment": "Negative", "justification": "Apreciada de forma negativa."} |

## Observações

- O script realiza a análise de sentimentos para cada comentário presente no arquivo CSV. Ele pode ser adaptado para analisar outros tipos de dados.
- Se o modelo não retornar um JSON válido, o script tentará aplicar heurísticas simples para categorizar o comentário.
- Em arquivos com muitos comentários, a análise pode demorar, dependendo da capacidade da sua máquina e do tamanho do modelo.

