"""
Treinamento com MLflow — Campeonato Brasileiro.

Modelo 1 (resultado): 5 experimentos — multiclasse mandante/empate/visitante.
Modelo 2 (cartão vermelho): 3 experimentos — binário desbalanceado.
Uso: python -m src.train
"""

from __future__ import annotations

import os
import sys

# MLflow imprime emojis — força UTF-8 para evitar UnicodeEncodeError no Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.pipeline import (
    TARGET_CARTAO,
    TARGET_RESULTADO,
    build_preprocessor,
    encode_target_resultado,
    get_feature_cols,
    save_pipeline,
)

MLFLOW_URI = "http://localhost:5000"
RANDOM_STATE = 42
TEST_SIZE = 0.20

MODELOS_RESULTADO = {
    "logistic_regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    ),
    "xgboost_default": XGBClassifier(
        n_estimators=200, objective="multi:softprob",
        eval_metric="mlogloss", random_state=RANDOM_STATE, verbosity=0,
    ),
    "xgboost_tuned": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=RANDOM_STATE, verbosity=0,
    ),
}

MODELOS_CARTAO = {
    "logistic_regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "xgboost": XGBClassifier(
        n_estimators=200, scale_pos_weight=3,
        eval_metric="aucpr", random_state=RANDOM_STATE, verbosity=0,
    ),
}


def _carregar_dados() -> pd.DataFrame:
    raw_files = sorted(Path("data/raw").glob("brasileirao_*.parquet"))
    if not raw_files:
        raise FileNotFoundError(
            "Nenhum arquivo encontrado. Execute 'python -m src.data_ingestion' primeiro."
        )
    df = pd.read_parquet(raw_files[-1])
    logger.info(f"Carregado {raw_files[-1].name} | {len(df)} partidas")
    return df


def _treinar_resultado(df: pd.DataFrame, feat_cols: list[str]) -> Pipeline:
    """Treina 5 modelos de resultado, loga no MLflow, retorna o melhor."""
    mlflow.set_experiment("brasileirao-resultado")

    y_raw = df[TARGET_RESULTADO]
    y, le = encode_target_resultado(y_raw)
    X = df[feat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"[resultado] treino={len(X_train)} | teste={len(X_test)}")
    logger.info(f"Classes: {dict(zip(le.classes_, range(len(le.classes_))))}")

    best_f1 = -1.0
    best_pipeline = None

    for nome, modelo in MODELOS_RESULTADO.items():
        preprocessor = build_preprocessor(feat_cols)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", modelo)])

        with mlflow.start_run(run_name=f"resultado_{nome}"):
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metricas = {
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
                "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
                "accuracy": float((y_pred == y_test).mean()),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": len(feat_cols),
            }

            mlflow.log_params({"model_type": nome, "target": "resultado"})
            mlflow.log_metrics(metricas)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            logger.info(
                f"  {nome:25s} | F1-macro={metricas['f1_macro']:.4f} | "
                f"Acc={metricas['accuracy']:.4f}"
            )

            if metricas["f1_macro"] > best_f1:
                best_f1 = metricas["f1_macro"]
                best_pipeline = pipeline

    logger.success(f"[resultado] Melhor F1-macro={best_f1:.4f}")
    return best_pipeline, le


def _treinar_cartao(df: pd.DataFrame, feat_cols: list[str]) -> Pipeline:
    """Treina 3 modelos de cartão vermelho, loga no MLflow, retorna o melhor."""
    mlflow.set_experiment("brasileirao-cartao-vermelho")

    y = df[TARGET_CARTAO]
    X = df[feat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        f"[cartão vermelho] treino={len(X_train)} | teste={len(X_test)} | "
        f"prevalência={y_test.mean():.1%}"
    )

    best_auc_pr = -1.0
    best_pipeline = None

    for nome, modelo in MODELOS_CARTAO.items():
        preprocessor = build_preprocessor(feat_cols)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", modelo)])

        with mlflow.start_run(run_name=f"cartao_{nome}"):
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            metricas = {
                "roc_auc": roc_auc_score(y_test, y_proba),
                "auc_pr": average_precision_score(y_test, y_proba),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "prevalencia_positivos": float(y_test.mean()),
            }

            mlflow.log_params({"model_type": nome, "target": "cartao_vermelho"})
            mlflow.log_metrics(metricas)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            logger.info(
                f"  {nome:25s} | AUC-PR={metricas['auc_pr']:.4f} | "
                f"ROC-AUC={metricas['roc_auc']:.4f}"
            )

            if metricas["auc_pr"] > best_auc_pr:
                best_auc_pr = metricas["auc_pr"]
                best_pipeline = pipeline

    logger.success(f"[cartão vermelho] Melhor AUC-PR={best_auc_pr:.4f}")
    return best_pipeline


def treinar_todos():
    mlflow.set_tracking_uri(MLFLOW_URI)

    df = _carregar_dados()
    feat_cols = get_feature_cols(df)
    logger.info(f"Features rolantes: {len(feat_cols)} colunas")

    logger.info("=== Modelo 1: Resultado da partida (5 experimentos) ===")
    pipeline_resultado, le_resultado = _treinar_resultado(df, feat_cols)

    logger.info("=== Modelo 2: Cartão vermelho (3 experimentos) ===")
    pipeline_cartao = _treinar_cartao(df, feat_cols)

    Path("models").mkdir(exist_ok=True)
    save_pipeline(
        {
            "pipeline_resultado": pipeline_resultado,
            "pipeline_cartao": pipeline_cartao,
            "label_encoder_resultado": le_resultado,
            "feature_cols": feat_cols,
        }
    )
    logger.success("Artefatos salvos em models/model_final.joblib")


if __name__ == "__main__":
    treinar_todos()
