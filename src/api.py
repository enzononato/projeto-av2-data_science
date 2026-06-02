"""
API de serving — recebe estatísticas rolantes de uma partida e retorna:
  - resultado previsto (mandante / empate / visitante) com probabilidades
  - probabilidade de cartão vermelho na partida

Uso:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
    docker compose up

Exemplo de chamada:
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"mandante": "Flamengo", "visitante": "Palmeiras",
           "features": {"mandante_chutes": 14.2, "visitante_chutes": 11.0}}'
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model_final.joblib"))

app = FastAPI(
    title="Campeonato Brasileiro — Predição de Partidas",
    description=(
        "Prediz resultado (mandante/empate/visitante) e probabilidade de "
        "cartão vermelho com base em médias históricas dos times."
    ),
    version="0.1.0",
)

_artifact = None


def _carregar_modelo() -> dict:
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Modelo não encontrado em {MODEL_PATH}. "
                "Execute 'python -m src.train'."
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


class PartidaInput(BaseModel):
    """
    Features rolantes (médias das últimas 5 partidas) de cada time.
    Valores ausentes podem ser omitidos — o modelo imputa com a mediana de treino.
    """

    mandante: str = Field(..., description="Nome do time mandante", examples=["Flamengo"])
    visitante: str = Field(..., description="Nome do time visitante", examples=["Palmeiras"])
    features: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Features rolantes no formato {nome_feature: valor}. "
            "Nomes disponíveis: mandante_chutes, mandante_gols_marcados, visitante_chutes, etc."
        ),
        examples=[{
            "mandante_chutes": 14.2,
            "mandante_chutes_no_alvo": 5.4,
            "mandante_gols_marcados": 1.8,
            "visitante_chutes": 11.0,
            "visitante_gols_marcados": 1.2,
        }],
    )


class PredicaoOutput(BaseModel):
    mandante: str
    visitante: str
    resultado_previsto: str
    probabilidades: dict[str, float]
    prob_cartao_vermelho: float
    interpretacao: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredicaoOutput)
def predict(payload: PartidaInput):
    artifact = _carregar_modelo()
    pipeline_resultado = artifact["pipeline_resultado"]
    pipeline_cartao = artifact["pipeline_cartao"]
    le = artifact["label_encoder_resultado"]
    feat_cols: list[str] = artifact["feature_cols"]

    row = {col: payload.features.get(col, float("nan")) for col in feat_cols}
    df = pd.DataFrame([row])

    try:
        proba_resultado = pipeline_resultado.predict_proba(df)[0]
        resultado_idx = int(proba_resultado.argmax())
        resultado = le.classes_[resultado_idx]

        prob_cv = float(pipeline_cartao.predict_proba(df)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {exc}")

    probas_dict = {cls: round(float(p), 4) for cls, p in zip(le.classes_, proba_resultado)}

    if resultado == "mandante":
        interp = f"Favorito: {payload.mandante} (probabilidade {probas_dict['mandante']:.0%})"
    elif resultado == "visitante":
        interp = f"Favorito: {payload.visitante} (probabilidade {probas_dict['visitante']:.0%})"
    else:
        interp = f"Empate previsto (probabilidade {probas_dict['empate']:.0%})"

    if prob_cv >= 0.50:
        interp += " | Alta chance de cartão vermelho na partida."

    return PredicaoOutput(
        mandante=payload.mandante,
        visitante=payload.visitante,
        resultado_previsto=resultado,
        probabilidades=probas_dict,
        prob_cartao_vermelho=round(prob_cv, 4),
        interpretacao=interp,
    )
