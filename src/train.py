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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
CV_SPLITS = 5  # TimeSeriesSplit folds para Grid Search

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

# ---------------------------------------------------------------------------
# Grades de hiperparâmetros para GridSearchCV
# Chaves no formato "model__<param>" para corresponder ao step "model" do Pipeline.
# Grades pequenas para treino rápido; suficientes para demonstrar tuning acadêmico.
# ---------------------------------------------------------------------------

GRIDS_RESULTADO = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
    },
    "random_forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, None],
    },
    "gradient_boosting": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 4],
    },
    "xgboost_default": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 6],
    },
    "xgboost_tuned": {
        "model__n_estimators": [200, 300],
        "model__learning_rate": [0.03, 0.05],
    },
}

GRIDS_CARTAO = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
    },
    "random_forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, None],
    },
    "xgboost": {
        "model__n_estimators": [100, 200],
        "model__scale_pos_weight": [3, 5, 10],
    },
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


def _treinar_resultado(df: pd.DataFrame, feat_cols: list[str]) -> tuple:
    """Treina 5 modelos de resultado com Grid Search, loga no MLflow, retorna o melhor."""
    mlflow.set_experiment("brasileirao-resultado")

    y_raw = df[TARGET_RESULTADO]
    y, le = encode_target_resultado(y_raw)

    X_train, X_test, y_train, y_test, data_corte = _split_cronologico(df, feat_cols, y)
    logger.info(f"[resultado] treino={len(X_train)} | teste={len(X_test)}")
    logger.info(f"Classes: {dict(zip(le.classes_, range(len(le.classes_))))}")

    best_f1 = -1.0
    best_pipeline = None
    cv = TimeSeriesSplit(n_splits=CV_SPLITS)

    for nome, modelo in MODELOS_RESULTADO.items():
        preprocessor = build_preprocessor(feat_cols)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", modelo)])

        with mlflow.start_run(run_name=f"resultado_{nome}"):
            t0 = time.time()

            gs = GridSearchCV(
                pipeline,
                param_grid=GRIDS_RESULTADO[nome],
                cv=cv,
                scoring="f1_macro",
                n_jobs=1,
                refit=True,
                verbose=0,
                error_score=0.0,
            )
            gs.fit(X_train, y_train)
            pipeline = gs.best_estimator_
            tempo_treino = round(time.time() - t0, 2)

            y_pred = pipeline.predict(X_test)
            report = classification_report(
                y_test, y_pred, target_names=le.classes_, output_dict=True
            )

            metricas: dict = {
                "f1_macro":         f1_score(y_test, y_pred, average="macro"),
                "f1_weighted":      f1_score(y_test, y_pred, average="weighted"),
                "accuracy":         float((y_pred == y_test).mean()),
                "best_cv_f1_macro": round(gs.best_score_, 4),
                "n_train":          len(X_train),
                "n_test":           len(X_test),
                "n_features":       len(feat_cols),
                "tempo_treino_s":   tempo_treino,
            }
            for classe in le.classes_:
                for metrica in ("precision", "recall", "f1-score"):
                    chave = f"{metrica.replace('-score', '')}_{classe}"
                    metricas[chave] = round(report[classe][metrica], 4)

            best_params = {
                f"gs_{k.replace('model__', '')}": v
                for k, v in gs.best_params_.items()
            }
            mlflow.log_params({
                "model_type": nome, "target": "resultado",
                "data_corte_teste": data_corte, "cv_splits": CV_SPLITS,
                **best_params,
            })
            mlflow.log_metrics(metricas)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            logger.info(
                f"  {nome:25s} | F1-macro={metricas['f1_macro']:.4f} | "
                f"CV-F1={metricas['best_cv_f1_macro']:.4f} | "
                f"best={gs.best_params_} | {tempo_treino}s"
            )

            if metricas["f1_macro"] > best_f1:
                best_f1 = metricas["f1_macro"]
                best_pipeline = pipeline

    logger.success(f"[resultado] Melhor F1-macro no teste={best_f1:.4f}")
    return best_pipeline, le


def _treinar_cartao(df: pd.DataFrame, feat_cols: list[str]) -> Pipeline:
    """Treina 3 modelos de cartão vermelho com Grid Search, loga no MLflow, retorna o melhor."""
    mlflow.set_experiment("brasileirao-cartao-vermelho")

    y = df[TARGET_CARTAO]

    X_train, X_test, y_train, y_test, data_corte = _split_cronologico(df, feat_cols, y)
    logger.info(
        f"[cartão vermelho] treino={len(X_train)} | teste={len(X_test)} | "
        f"prevalência={y_test.mean():.1%}"
    )

    best_auc_pr = -1.0
    best_pipeline = None
    cv = TimeSeriesSplit(n_splits=CV_SPLITS)

    for nome, modelo in MODELOS_CARTAO.items():
        preprocessor = build_preprocessor(feat_cols)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", modelo)])

        with mlflow.start_run(run_name=f"cartao_{nome}"):
            t0 = time.time()

            gs = GridSearchCV(
                pipeline,
                param_grid=GRIDS_CARTAO[nome],
                cv=cv,
                scoring="average_precision",
                n_jobs=1,
                refit=True,
                verbose=0,
                error_score=0.0,
            )
            gs.fit(X_train, y_train)
            pipeline = gs.best_estimator_
            tempo_treino = round(time.time() - t0, 2)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            report = classification_report(
                y_test, y_pred,
                target_names=["sem_vermelho", "com_vermelho"],
                output_dict=True,
            )

            metricas: dict = {
                "roc_auc":               roc_auc_score(y_test, y_proba),
                "auc_pr":                average_precision_score(y_test, y_proba),
                "brier_score":           round(brier_score_loss(y_test, y_proba), 4),
                "f1_macro":              f1_score(y_test, y_pred, average="macro"),
                "precision_cv":          round(report["com_vermelho"]["precision"], 4),
                "recall_cv":             round(report["com_vermelho"]["recall"], 4),
                "f1_cv":                 round(report["com_vermelho"]["f1-score"], 4),
                "best_cv_auc_pr":        round(gs.best_score_, 4),
                "n_train":               len(X_train),
                "n_test":                len(X_test),
                "prevalencia_positivos": float(y_test.mean()),
                "tempo_treino_s":        tempo_treino,
            }

            best_params = {
                f"gs_{k.replace('model__', '')}": v
                for k, v in gs.best_params_.items()
            }
            mlflow.log_params({
                "model_type": nome, "target": "cartao_vermelho",
                "data_corte_teste": data_corte, "cv_splits": CV_SPLITS,
                **best_params,
            })
            mlflow.log_metrics(metricas)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            logger.info(
                f"  {nome:25s} | AUC-PR={metricas['auc_pr']:.4f} | "
                f"CV-AUC-PR={metricas['best_cv_auc_pr']:.4f} | "
                f"best={gs.best_params_} | {tempo_treino}s"
            )

            if metricas["auc_pr"] > best_auc_pr:
                best_auc_pr = metricas["auc_pr"]
                best_pipeline = pipeline

    logger.success(f"[cartão vermelho] Melhor AUC-PR no teste={best_auc_pr:.4f}")
    return best_pipeline


def treinar_todos():
    mlflow.set_tracking_uri(MLFLOW_URI)

    df = _carregar_dados()
    feat_cols = get_feature_cols(df)
    logger.info(f"Features rolantes: {len(feat_cols)} colunas")

    logger.info("=== Modelo 1: Resultado da partida (5 experimentos + Grid Search) ===")
    pipeline_resultado, le_resultado = _treinar_resultado(df, feat_cols)

    logger.info("=== Modelo 2: Cartão vermelho (3 experimentos + Grid Search) ===")
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
