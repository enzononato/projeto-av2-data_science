# Predição de Resultados e Cartões Vermelhos no Campeonato Brasileiro de Futebol

> **RASCUNHO** — base para o artigo final (máx. 5 páginas). Preencham os campos marcados com `[...]`,
> revisem o texto com a dupla e exportem para PDF em `reports/artigo.pdf`.
>
> **Autores:** [Nome completo de cada integrante] — FACAPE, Ciência de Dados Aplicada
> **Professor:** Mateus Silva | **Avaliação:** AV2

---

## 1. Introdução

O futebol movimenta decisões de clubes, casas de análise esportiva e portais de estatística.
Antes de cada partida, esses agentes precisam de **estimativas probabilísticas** — qual a chance
de vitória do mandante, de empate ou de vitória do visitante — e do **risco de incidentes
disciplinares**, como cartões vermelhos, que afetam escalação e planejamento tático.

**Problema.** Prever, **antes do apito inicial**, (1) o resultado de uma partida do Campeonato
Brasileiro (mandante / empate / visitante) e (2) a probabilidade de a partida ter ao menos um
cartão vermelho, usando apenas o histórico recente de cada time.

**Stakeholder hipotético.** Departamentos de análise de desempenho de clubes e portais
esportivos, que usam essas probabilidades para apoiar decisões de escalação e narrativa.

**Métrica de sucesso.** Para o resultado, **F1-macro** (trata as três classes com igual peso,
mesmo o empate, que é minoritário). Para o cartão vermelho, **AUC-PR** (apropriada para classe
rara) comparada ao baseline da prevalência.

**Objetivo.** Construir um pipeline reprodutível, sem vazamento de dados (*leakage*), que treine
e compare múltiplos modelos, rastreie experimentos e seja servido por uma API containerizada.

---

## 2. Dados

**Fonte.** Campeonato Brasileiro de Futebol, via Kaggle (`adaoduque/campeonato-brasileiro-de-futebol`),
cobrindo as temporadas de **2003 a 2025**.

**Volume.** Após a engenharia de features, **9.138 partidas × 31 colunas**. O dataset bruto reúne
quatro arquivos (partidas, estatísticas por clube, cartões e gols), totalizando dezenas de milhares
de registros antes da agregação por partida.

**Alvos.**
- `resultado` — multiclasse: `mandante`, `empate`, `visitante`.
- `tem_cartao_vermelho` — binário: 1 se a partida teve ao menos um cartão vermelho.

**Análise exploratória (EDA).** O notebook `notebooks/01_eda.ipynb` gera as figuras em `reports/`:
- `fig_partidas_por_ano.png` — volume de partidas por temporada.
- `fig_target_resultado.png` — distribuição dos resultados (vantagem do mandante).
- `fig_target_cartao.png` — distribuição do cartão vermelho.
- `fig_boxplots_features.png` — dispersão das estatísticas.
- `fig_correlacao.png` — correlação entre features.
- `fig_aproveitamento_times.png` — aproveitamento histórico por time.

**Achado crítico de qualidade de dados.** A estatística de cartão vermelho **só foi registrada
de forma sistemática a partir de 2015** (e há uma lacuna em 2024). Nas temporadas de 2003 a 2014,
a coluna aparece zerada — não por ausência de cartões, mas por **ausência de registro**. Treinar
o modelo com esses "zeros falsos" corromperia o alvo. Por isso, **o modelo de cartão vermelho foi
restrito às temporadas com dado efetivamente registrado** (2015–2023 e 2025), totalizando
3.798 partidas com prevalência de ~21%. `[Inserir aqui fig_target_cartao.png como evidência]`

---

## 3. Metodologia (CRISP-DM)

O projeto segue as seis fases do CRISP-DM, com rastro no repositório.

**3.1 Engenharia de features e anti-leakage.**
As features são **médias móveis das últimas 5 partidas** de cada time (`shift(1)` + `rolling(5)`),
calculadas **antes** do jogo. Ou seja, as estatísticas reais da partida em questão (posse, chutes,
faltas etc.) **nunca entram como variáveis preditoras** — apenas o desempenho pregresso. Isso
elimina o vazamento de informação do futuro para o passado. São 24 features (12 estatísticas ×
mandante/visitante): chutes, chutes no alvo, faltas, gols marcados, gols sofridos, escanteios,
impedimentos, cartões amarelos, entre outras.

**3.2 Divisão treino/teste temporal (out-of-time).**
Em vez de divisão aleatória, usamos **corte cronológico**: os 80% de partidas mais antigas para
treino e os 20% mais recentes para teste. Isso simula o uso real (treinar no passado, prever o
futuro) e evita o vazamento que ocorreria ao misturar épocas.

**3.3 Modelagem e busca de hiperparâmetros.**
Para o **resultado**, comparamos 5 modelos: Regressão Logística, Random Forest, Gradient Boosting
e XGBoost (configuração padrão e ajustada). Para o **cartão vermelho**, 3 modelos: Regressão
Logística, Random Forest e XGBoost (com `scale_pos_weight` para o desbalanceamento). Cada modelo
passou por **Grid Search com validação cruzada temporal** (`TimeSeriesSplit`, 5 folds), garantindo
que a busca de hiperparâmetros também respeite a ordem do tempo.

**3.4 Rastreabilidade.**
Todos os experimentos são registrados no **MLflow** (8 runs, acima do mínimo de 5 exigido), com
métricas, hiperparâmetros vencedores e o modelo serializado. O monitoramento de *data drift* usa
**Evidently** (`src/evaluate.py`).

**3.5 Pré-processamento.**
Pipeline `scikit-learn` com `SimpleImputer` (mediana) + `StandardScaler`, encapsulado junto ao
modelo para garantir que o scaler seja serializado e reaplicado de forma idêntica em produção.

---

## 4. Resultados

**4.1 Resultado da partida (multiclasse).**

| Modelo | F1-macro | Acurácia | F1 (CV temporal) |
|---|---|---|---|
| **Random Forest** | **0,377** | **0,415** | 0,280 |
| Logistic Regression | 0,355 | 0,355 | 0,296 |
| XGBoost (default) | 0,345 | 0,403 | 0,300 |
| XGBoost (tuned) | 0,343 | 0,437 | 0,268 |
| Gradient Boosting | 0,276 | 0,294 | 0,288 |

> Baseline (prever sempre mandante) ≈ 0,26 de F1-macro. Todos os modelos superam o acaso.
> O melhor modelo (Random Forest) atinge F1-macro de 0,377.

**4.2 Cartão vermelho (binário, prevalência 21,3%).**

| Modelo | AUC-PR | ROC-AUC | Brier |
|---|---|---|---|
| **Random Forest** | **0,243** | 0,543 | 0,168 |
| Logistic Regression | 0,228 | 0,528 | 0,235 |
| XGBoost | 0,224 | 0,540 | 0,202 |

> Baseline AUC-PR = 0,213 (prevalência). O melhor modelo fica ligeiramente acima do baseline.

**4.3 Análise de erros e interpretação de negócio.**
O resultado de futebol é **inerentemente imprevisível** a partir apenas de médias históricas: a
acurácia de ~41% supera o acaso, mas confirma que fatores não capturados (lesões, escalação,
fator casa específico, arbitragem) pesam muito. O empate é a classe mais difícil de prever, como
esperado. Para o cartão vermelho, o ganho modesto sobre o baseline indica que **estatísticas
agregadas pré-jogo têm baixo poder preditivo sobre incidentes disciplinares** — um achado honesto
que orienta trabalhos futuros (incorporar dados de arbitragem e histórico de confrontos diretos).
`[Opcional: incluir análise de desempenho por temporada gerada por src/evaluate.py]`

---

## 5. Conclusão e Trabalhos Futuros

Construímos um pipeline completo de Ciência de Dados — da ingestão ao serving containerizado —
seguindo o CRISP-DM, com rigor anti-leakage (features defasadas + split temporal) e rastreabilidade
via MLflow. O melhor modelo de resultado (Random Forest) atinge F1-macro de 0,377, superando o
baseline; o modelo de cartão vermelho fica modestamente acima da prevalência.

**Limitações.**
- O resultado de partidas tem teto de previsibilidade alto a partir de estatísticas agregadas.
- A cobertura da estatística de cartão vermelho começa só em 2015, reduzindo a amostra desse alvo.
- Não há dados de escalação, lesões ou arbitragem, que provavelmente explicariam parte da variância.

**Trabalhos futuros.**
- Adicionar features de confronto direto (head-to-head) e força relativa (Elo).
- Incorporar dados de arbitragem para o modelo de cartão vermelho.
- Calibrar probabilidades (Platt/Isotônica) e otimizar o threshold por custo de decisão.
- Monitoramento contínuo de drift em produção com alertas automáticos.

---

## 6. Referências

- Dataset: *Campeonato Brasileiro de Futebol* — Kaggle (`adaoduque/campeonato-brasileiro-de-futebol`).
- Pedregosa et al. *Scikit-learn: Machine Learning in Python*. JMLR, 2011.
- Chen & Guestrin. *XGBoost: A Scalable Tree Boosting System*. KDD, 2016.
- *MLflow: An open source platform for the machine learning lifecycle*. mlflow.org.
- *Evidently AI* — biblioteca de monitoramento de modelos. evidentlyai.com.
- Repositório do projeto: https://github.com/enzononato/projeto-av2-data_science (tag `v1.0`).
