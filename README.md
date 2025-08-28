# Projeto de Análise de Sentimentos em Comentários do YouTube  

Este projeto tem como finalidade a **análise de sentimentos em comentários de vídeos do YouTube**, integrando abordagens manuais e automáticas (LLM). O estudo resulta em **relatórios consolidados, representações gráficas e métricas de avaliação de desempenho (com ênfase no F1-score)**, a fim de oferecer uma visão clara e comparativa da acurácia entre os diferentes métodos.  

## Estrutura do Projeto  

```
projeto_sentimentos/
│── data/                         # Conjunto de dados brutos e processados
│   ├── comments/                 # Arquivos CSV originais
│   ├── analisados_manualmente/   # Anotações manuais
│   ├── analise_few_shots/        # Resultados obtidos via Few-Shot LLM
│   ├── analise_manual_few_shots/ # Revisões e ajustes manuais
│
│── src/                          # Código-fonte
│   ├── sentiment_analysis.py     # Script principal de análise
│   ├── analisar_csvs_ollama_few_shots.py # Análise utilizando Few-Shot LLM
│   ├── graficos.py               # Geração de representações gráficas automáticas
│   ├── graficos_manual.py        # Geração de gráficos a partir da análise manual
│
│── results/                      # Resultados gerados
│   ├── f1_score/                 # Métricas de avaliação
│   ├── graficos/                 # Gráficos oriundos da análise automática
│   ├── graficos_manual/          # Gráficos oriundos da análise manual
│   ├── resumo_agregado.csv       # Resumo consolidado das análises
│
│── docs/                         # Documentação
│   ├── README.md                 # Este documento
│
│── requirements.txt              # Dependências do projeto
```

## Funcionalidades  

- **Pré-processamento e organização de comentários** provenientes de arquivos CSV.  
- **Classificação de sentimentos** em três categorias:  
  - `Positive → 1`  
  - `Neutral → 0`  
  - `Negative → -1`  
- **Análise automática** empregando **LLM Few-Shot**.  
- **Análise manual** de subconjuntos de comentários para aferição da qualidade.  
- **Comparação de desempenho** entre métodos de análise.  
- **Cálculo de métricas de avaliação** (precisão, revocação e F1-score).  
- **Visualização dos resultados** por meio de gráficos (barras e setores).  

## Instruções de Execução  

### 1. Clonar o repositório  
```bash
git clone https://github.com/seuusuario/projeto_sentimentos.git
cd projeto_sentimentos
```

### 2. Instalar dependências  
```bash
pip install -r requirements.txt
```

### 3. Executar os scripts principais  
Análise automática:  
```bash
python src/sentiment_analysis.py
```

Análise via Few-Shot LLM:  
```bash
python src/analisar_csvs_ollama_few_shots.py
```

Geração de gráficos:  
```bash
python src/graficos.py
```

### 4. Resultados Obtidos  
- Métricas disponíveis em `results/f1_score/`.  
- Gráficos em `results/graficos/` e `results/graficos_manual/`.  
- Resumo consolidado em `results/resumo_agregado.csv`.  

## Exemplos de Saídas  

- Distribuição de sentimentos por vídeo.  
- Comparativo entre análise manual e automática.  
- Percentuais de cada categoria de sentimento (`Positive`, `Neutral`, `Negative`).  

- O projeto foi desenvolvido em **Python 3.10 ou superior**. 

## Autor  

Este projeto foi desenvolvido no âmbito de pesquisa acadêmica em **Análise de Sentimentos**, no Instituto Federal do Piauí – Campus Picos.  
