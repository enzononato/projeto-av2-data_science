"""
Pipeline de pré-processamento para o dataset do Campeonato Brasileiro.

Dois targets:
  resultado          — 'mandante' | 'empate' | 'visitante'  (multiclasse)
  tem_cartao_vermelho — 0 | 1  (binário, desbalanceado)

Features: médias rolantes das últimas 5 partidas de cada time (mandante_* e visitante_*).
Todas numéricas — sem features categóricas para evitar leakage de identidade dos clubes.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET_RESULTADO = "resultado"
TARGET_CARTAO = "tem_cartao_vermelho"

# Colunas que nunca são features (identificadores e targets)
ID_COLS = ["partida_id", "rodata", "data", "mandante", "visitante",
           "mandante_Estado", "visitante_Estado"]
TARGET_COLS = [TARGET_RESULTADO, TARGET_CARTAO]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Retorna colunas de features numéricas (exclui IDs e targets)."""
    return [
        c for c in df.select_dtypes(include="number").columns
        if c not in TARGET_COLS
    ]


def encode_target_resultado(y: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    """Codifica resultado → inteiro. Retorna série codificada + encoder."""
    le = LabelEncoder()
    return pd.Series(le.fit_transform(y), index=y.index), le


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """
    Retorna ColumnTransformer (não fitado) para uso no Pipeline sklearn.
    Todas as features são numéricas: imputa mediana + padroniza.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_cols)],
        remainder="drop",
    )


def save_pipeline(artifact: dict, path: Path = Path("models/model_final.joblib")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_pipeline(path: Path = Path("models/model_final.joblib")) -> dict:
    return joblib.load(path)


def load_dataset(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)
