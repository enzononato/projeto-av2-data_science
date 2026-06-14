# Predição de Resultados no Campeonato Brasileiro

Sistema de Machine Learning que prediz **resultado de partidas** (mandante/empate/visitante)
e **probabilidade de cartão vermelho**, usando o histórico do Campeonato Brasileiro de 2003 a 2025.

## 🚀 Início Rápido — Rodar SÓ com Docker (sem treinar nada)

Esta é a forma mais simples de rodar o projeto na sua máquina. **Não precisa** instalar Python,
nem configurar Kaggle, nem treinar nada — o modelo treinado e as runs do MLflow já vêm no repositório.

> ✅ **Você NÃO precisa treinar o modelo.** O arquivo `models/model_final.joblib` e o histórico do
> MLflow (`mlflow.db` + `mlartifacts/`) já estão versionados. Basta ter o Docker e rodar **um comando**.
> Treinar do zero (Kaggle + ingestão + `python -m src.train`) só é necessário se você quiser
> **regerar** os modelos — veja a seção [Como Rodar (desenvolvimento)](#como-rodar) mais abaixo.

### Pré-requisitos
- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e **aberto**
  (espere o ícone da baleia ficar estável antes de continuar)

### Passo a passo

**1. Clonar o repositório**
```bash
git clone https://github.com/enzononato/projeto-av2-data_science.git
cd projeto-av2-data_science
```

**2. Subir a aplicação (API + MLflow juntos)**
```bash
docker compose up --build
```
Na primeira vez demora alguns minutos (constrói a imagem da API e baixa o MLflow).
Deixe esse terminal aberto — é o servidor rodando. Use `docker compose up -d --build`
para rodar em segundo plano e liberar o terminal.

**3. Acessar no navegador**

| Serviço | URL | O que mostra |
|---|---|---|
| **API de predição** | http://localhost:8000/docs | Swagger interativo para testar `/predict` |
| **MLflow UI** | http://localhost:5000 | Os 8 experimentos comparados (5 resultado + 3 cartão) |

**4. Testar uma predição**

Em http://localhost:8000/docs → `POST /predict` → **Try it out** → cole o corpo abaixo → **Execute**:
```json
{
  "mandante": "Flamengo",
  "visitante": "Palmeiras",
  "features": {
    "mandante_chutes": 15.2,
    "mandante_gols_marcados": 2.0,
    "visitante_chutes": 12.4,
    "visitante_gols_marcados": 1.6
  }
}
```

**5. Parar a aplicação**
```bash
# Ctrl+C no terminal (se rodou em primeiro plano), depois:
docker compose down
```

> **Onde estão os modelos no MLflow?** No MLflow 3.x, os modelos aparecem na aba **Models**
> de cada run (não na aba *Artifacts*). São 8 modelos logados, um por experimento.

---

## Resultados Principais

> Métricas obtidas com **Grid Search (TimeSeriesSplit, 5 folds)** e avaliação **out-of-time**
> (treino nos 80% mais antigos, teste nos 20% mais recentes).

### Modelo 1 — Resultado da Partida (multiclasse)

| Modelo | F1-macro | Acurácia |
|---|---|---|
| **Random Forest** | **0.3767** | 0.4147 |
| Logistic Regression | 0.3548 | 0.3550 |
| XGBoost (default) | 0.3453 | 0.4026 |
| XGBoost (tuned) | 0.3428 | 0.4365 |
| Gradient Boosting | 0.2763 | 0.2943 |

> Baseline (sempre prever mandante): F1-macro ≈ 0.26 — todos os modelos superam o baseline.

### Modelo 2 — Cartão Vermelho (binário, 21.3% prevalência)

Restrito às temporadas com registro real de cartão vermelho (2015–2023 e 2025).

| Modelo | AUC-PR | ROC-AUC | Brier |
|---|---|---|---|
| **Random Forest** | **0.2425** | 0.5428 | 0.1676 |
| Logistic Regression | 0.2277 | 0.5279 | 0.2354 |
| XGBoost | 0.2243 | 0.5399 | 0.2022 |

> Baseline AUC-PR = 0.213 (prevalência). Prever cartão a partir de estatísticas pré-jogo é
> genuinamente difícil — o ganho modesto sobre o baseline é um resultado honesto.

## Contexto de Negócio

**Problema:** Clubes, analistas esportivos e plataformas de dados precisam de estimativas
probabilísticas para cada partida antes do jogo, com base no desempenho recente das equipes.

**Stakeholder:** Departamentos de análise de desempenho de clubes, portais esportivos, ligas

**Decisão suportada:** Probabilidade de vitória do mandante / empate / vitória do visitante
e risco de partida com cartão vermelho (gestão de escalação, planejamento tático)

## Dataset

- **Fonte:** Campeonato Brasileiro via Kaggle (`adaoduque/campeonato-brasileiro-de-futebol`)
- **Período:** 2003 – 2025 (22 temporadas)
- **Partidas brutas:** 9.165
- **Após engenharia de features:** 9.138 partidas × 31 colunas
- **Arquivos usados:**
  - `campeonato-brasileiro-full.csv` — resultado, clubes, formação
  - `campeonato-brasileiro-estatisticas-full.csv` — chutes, posse, passes, faltas, cartões

## Anti-Leakage

Features = **médias rolantes das últimas 5 partidas** de cada time, calculadas **antes** do jogo.
Estatísticas do jogo atual (posse real, chutes reais) **nunca entram como features**.

## Estrutura do Projeto

```
├── pyproject.toml          # Dependências Python (PEP 517/518)
├── Dockerfile              # Containerização da API
├── docker-compose.yml      # API + MLflow como serviços
├── .env.example            # Template de variáveis de ambiente
├── data/raw/               # Parquets gerados pela ingestão (gitignored)
├── notebooks/
│   └── 01_eda.ipynb        # Análise exploratória
├── src/
│   ├── data_ingestion.py   # Kagglehub + rolling features (anti-leakage)
│   ├── pipeline.py         # Targets, features e preprocessor
│   ├── train.py            # 5 + 3 experimentos MLflow (resultado + cartão)
│   ├── evaluate.py         # Análise por temporada + drift (Evidently)
│   └── api.py              # FastAPI — /predict retorna resultado + cartão vermelho
├── models/                 # Modelo serializado (commitado para Docker funcionar)
└── reports/                # Figuras do EDA e relatórios
```

## Como Rodar

### 1. Instalar dependências

```bash
python -m venv venv
venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configurar Kaggle

Crie a conta em kaggle.com, gere um API token em *Account → API → Create New Token*
e salve em `C:\Users\<seu_usuario>\.kaggle\kaggle.json`.

### 3. Coletar e preparar dados

```bash
python -m src.data_ingestion
```

Baixa o dataset, computa médias rolantes por time e salva em `data/raw/brasileirao_<timestamp>.parquet`.

### 4. Treinar modelos (MLflow deve estar rodando)

Terminal 1:
```bash
venv\Scripts\python -m mlflow ui --host 0.0.0.0 --port 5000
```

Terminal 2:
```bash
python -m src.train
```

Acesse os experimentos em http://localhost:5000 — experimentos `brasileirao-resultado` e `brasileirao-cartao-vermelho`.

### 5. Subir a API de predição

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

**Exemplo de requisição:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mandante": "Flamengo",
    "visitante": "Palmeiras",
    "features": {
      "mandante_chutes": 15.2,
      "mandante_gols_marcados": 2.0,
      "mandante_gols_sofridos": 0.8,
      "visitante_chutes": 12.4,
      "visitante_gols_marcados": 1.6,
      "visitante_gols_sofridos": 1.1
    }
  }'
```

**Resposta:**
```json
{
  "mandante": "Flamengo",
  "visitante": "Palmeiras",
  "resultado_previsto": "mandante",
  "probabilidades": {"empate": 0.27, "mandante": 0.52, "visitante": 0.21},
  "prob_cartao_vermelho": 0.09,
  "interpretacao": "Favorito: Flamengo (probabilidade 52%)"
}
```

### 6. Rodar via Docker

Sobe a API (porta 8000) e o MLflow UI (porta 5000) com um único comando:

```bash
docker compose up --build
```

> Passo a passo completo (clone + execução) na seção [🚀 Início Rápido](#-início-rápido--rodar-via-docker-clone-do-github) no topo deste README.

## Metodologia (CRISP-DM)

1. **Business Understanding** — Predição pré-jogo para suporte a decisões táticas
2. **Data Understanding** — EDA em `notebooks/01_eda.ipynb`
3. **Data Preparation** — `src/data_ingestion.py`: rolling features (últimas 5 partidas, shift+rolling)
4. **Modeling** — `src/train.py`: 5 modelos para resultado + 3 para cartão vermelho
5. **Evaluation** — `src/evaluate.py`: métricas por temporada + drift Evidently
6. **Deployment** — FastAPI containerizada com Docker

### Prevenção de Data Leakage

O projeto usa apenas o **histórico pregresso** de cada time como features.
A função `_rolling_por_clube()` aplica `shift(1)` antes da janela rolante,
garantindo que a partida atual não contamina suas próprias features.

## Projeto Acadêmico

Disciplina: Ciência de Dados Aplicada — FACAPE  
Professor: Mateus Silva  
Avaliação: AV2
