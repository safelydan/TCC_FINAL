# Projeto de Análise de Sentimentos em Comentários do YouTube  

Este projeto tem como finalidade a **análise de sentimentos em comentários de vídeos do YouTube**, integrando abordagens **manuais e automáticas** com **Modelos de Linguagem de Grande Porte (LLMs)**.  
O estudo gera **relatórios consolidados, representações gráficas e métricas de avaliação de desempenho (com ênfase no F1-score)**, permitindo comparar a acurácia entre diferentes métodos de análise.  

---

## 📂 Estrutura do Projeto  

```
LLM_FINAL/
│── data/                         
│   ├── comments/                  # Arquivos CSV originais extraídos do YouTube
│   ├── analisados_manualmente/    # Conjunto de dados anotados manualmente
│   ├── analisados_few_shot/       # Resultados da classificação Few-Shot com LLMs
│   ├── analisados_manualmente_few_shot/ # Ajustes e revisões combinadas
│   ├── VADER/                     # Resultados da análise via VADER (baseline léxico)
│   └── f1/                        # Relatórios e métricas F1 consolidadas
│
│── src/                          
│   ├── sentiment_analysis.py                 # Script principal de análise
│   ├── analisar_csvs_ollama_few_shots.py     # Classificação usando Few-Shot via Ollama
│   ├── avaliar_resultados.py                 # Avaliação das métricas (precisão, revocação, F1)
│   ├── gerar_graficos.py                     # Geração de gráficos comparativos
│   └── utils.py                              # Funções auxiliares
│
│── requirements.txt              # Dependências do projeto
│── README.md                     # Documentação
```

---

## Como Executar  

1. **Clonar o repositório**  
   ```bash
   git clone https://github.com/seuusuario/LLM_FINAL.git
   cd LLM_FINAL
   ```

2. **Criar ambiente virtual e instalar dependências**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows

   pip install -r requirements.txt
   ```

3. **Executar análise automática (LLM Few-Shot via Ollama)**  
   ```bash
   python src/analisar_csvs_ollama_few_shots.py
   ```

4. **Executar avaliação e gerar métricas**  
   ```bash
   python src/avaliar_resultados.py
   ```

5. **Gerar gráficos comparativos**  
   ```bash
   python src/gerar_graficos.py
   ```

---

## Resultados  

O projeto gera:  
- **Tabelas comparativas** entre anotações manuais, VADER e LLMs.  
- **Métricas de desempenho**: precisão, revocação e F1-score por classe (positivo, neutro, negativo).  
- **Gráficos** em formato `.png` mostrando comparações entre abordagens.  

---

## Tecnologias Utilizadas  

- **Python 3.10+**  
- **Bibliotecas principais**:  
  - `pandas`, `numpy` – manipulação de dados  
  - `scikit-learn` – métricas de avaliação  
  - `matplotlib`, `seaborn` – geração de gráficos  
  - `yt-dlp` – coleta de comentários (se aplicável)  
- **LLM**: Execução via **Ollama** para classificação automática  

---

## Objetivo Acadêmico  

Este repositório faz parte de um **Trabalho de Conclusão de Curso (TCC)** em **Análise e Desenvolvimento de Sistemas (ADS)**, cujo objetivo é avaliar metodologias distintas de **análise de sentimentos em mídias sociais**, destacando vantagens, limitações e potencial de uso em contextos acadêmicos e mercadológicos.  
