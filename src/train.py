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

import time

import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    roc_auc_score,
)
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


def _split_cronologico(
    df: pd.DataFrame, feat_cols: list[str], target: pd.Series
) -> tuple:
    """
    Divisão treino/teste respeitando a ordem temporal das partidas.
    Treina nos 80% mais antigos, testa nos 20% mais recentes.
    Garante que o modelo nunca vê o futuro durante o treino.
    """
    df_ord = df.sort_values("data").copy()
    target_ord = target.loc[df_ord.index]

    split_idx = int(len(df_ord) * (1 - TEST_SIZE))
    df_train = df_ord.iloc[:split_idx]
    df_test  = df_ord.iloc[split_idx:]

    data_corte = df_ord["data"].iloc[split_idx].date()
    logger.info(
        f"  Split cronológico: treino até {df_ord['data'].iloc[split_idx - 1].date()} "
        f"| teste a partir de {data_corte}"
    )

    return (
        df_train[feat_cols],
        df_test[feat_cols],
        target_ord.iloc[:split_idx],
        target_ord.iloc[split_idx:],
        str(data_corte),
    )


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

    X_train, X_test, y_train, y_test, data_corte = _split_cronologico(df, feat_cols, y)
    logger.info(f"[resultado] treino={len(X_train)} | teste={len(X_test)}")
    logger.info(f"Classes: {dict(zip(le.classes_, range(len(le.classes_))))}")

    best_f1 = -1.0
    best_pipeline = None

    PARAM_KEYS = ("n_estimators", "max_depth", "learning_rate",
                  "subsample", "colsample_bytree", "C", "max_iter")

    for nome, modelo in MODELOS_RESULTADO.items():
        preprocessor = build_preprocessor(feat_cols)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", modelo)])

        with mlflow.start_run(run_name=f"resultado_{nome}"):
            t0 = time.time()
            pipeline.fit(X_train, y_train)
            tempo_treino = round(time.time() - t0, 2)

            y_pred = pipeline.predict(X_test)
            report = classification_report(y_test, y_pred,
                                           target_names=le.classes_,
                                           output_dict=True)

            metricas = {
                "f1_macro":    f1_score(y_test, y_pred, average="macro"),
                "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
                "accuracy":    float((y_pred == y_test).mean()),
                "n_train":     len(X_train),
                "n_test":      len(X_test),
                "n_features":  len(feat_cols),
                "tempo_treino_s": tempo_treino,
            }
            # Métricas por classe (empate, mandante, visitante)
            for classe in le.classes_:
                for metrica in ("precision", "recall", "f1-score"):
                    chave = f"{metrica.replace('-score','')}_{classe}"
                    metricas[chave] = round(report[classe][metrica], 4)

            # Hiperparâmetros do modelo como params
            hparams = {k: v for k, v in modelo.get_params().items()
                       if k in PARAM_KEYS and v is not None}
            mlflow.log_params({"model_type": nome, "target": "resultado",
                               "data_corte_teste": data_corte, **hparams})
            mlflow.log_metrics(metricas)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            logger.info(
                f"  {nome:25s} | F1-macro={metricas['f1_macro']:.4f} | "
                f"Acc={metricas['accuracy']:.4f} | {tempo_treino}s"
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

    X_train, X_test, y_train, y_test, data_corte = _split_cronologico(df, feat_cols, y)
    logger.info(
        f"[cartão vermelho] treino={len(X_train)} | teste={len(X_test)} | "
        f"prevalência={y_test.mean():.1%}"
    )

    best_auc_pr = -1.0
    best_pipeline = None

    PARAM_KEYS = ("n_estimators", "max_depth", "learning_rate",
                  "subsample", "colsample_bytree", "scale_pos_weight", "C", "max_iter")

    for nome, modelo in MODELOS_CARTAO.items():
        preprocessor = build_preprocessor(feat_cols)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", modelo)])

        with mlflow.start_run(run_name=f"cartao_{nome}"):
            t0 = time.time()
            pipeline.fit(X_train, y_train)
            tempo_treino = round(time.time() - t0, 2)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            report = classification_report(y_test, y_pred,
                                           target_names=["sem_vermelho", "com_vermelho"],
                                           output_dict=True)

            metricas = {
                "roc_auc":              roc_auc_score(y_test, y_proba),
                "auc_pr":               average_precision_score(y_test, y_proba),
                "brier_score":          round(brier_score_loss(y_test, y_proba), 4),
                "f1_macro":             f1_score(y_test, y_pred, average="macro"),
                "precision_cv":         round(report["com_vermelho"]["precision"], 4),
                "recall_cv":            round(report["com_vermelho"]["recall"], 4),
                "f1_cv":                round(report["com_vermelho"]["f1-score"], 4),
                "n_train":              len(X_train),
                "n_test":               len(X_test),
                "prevalencia_positivos": float(y_test.mean()),
                "tempo_treino_s":       tempo_treino,
            }

            hparams = {k: v for k, v in modelo.get_params().items()
                       if k in PARAM_KEYS and v is not None}
            mlflow.log_params({"model_type": nome, "target": "cartao_vermelho",
                               "data_corte_teste": data_corte, **hparams})
            mlflow.log_metrics(metricas)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            logger.info(
                f"  {nome:25s} | AUC-PR={metricas['auc_pr']:.4f} | "
                f"ROC-AUC={metricas['roc_auc']:.4f} | {tempo_treino}s"
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
