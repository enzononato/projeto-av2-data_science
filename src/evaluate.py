"""
Avaliação de erros por segmento e detecção de drift (Evidently AI).

Uso: python -m src.evaluate
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report

from src.pipeline import TARGET_CARTAO, TARGET_RESULTADO, get_feature_cols

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def avaliar_resultado(df: pd.DataFrame, pipeline, le, feat_cols: list[str]) -> None:
    """Avalia modelo de resultado por temporada (ano)."""
    X = df[feat_cols]
    y_true_enc, _ = __import__("src.pipeline", fromlist=["encode_target_resultado"]).encode_target_resultado(df[TARGET_RESULTADO])
    y_pred = pipeline.predict(X)

    logger.info("\n" + classification_report(
        y_true_enc, y_pred, target_names=le.classes_
    ))

    # Segmentação por temporada
    if "data" in df.columns:
        df = df.copy()
        df["ano"] = pd.to_datetime(df["data"], errors="coerce").dt.year
        resultados = []
        for ano, grupo in df.groupby("ano"):
            y_t, _ = __import__("src.pipeline", fromlist=["encode_target_resultado"]).encode_target_resultado(grupo[TARGET_RESULTADO])
            y_p = pipeline.predict(grupo[feat_cols])
            resultados.append({
                "ano": ano,
                "n_partidas": len(grupo),
                "acuracia": float((y_p == y_t).mean()),
            })
        seg = pd.DataFrame(resultados)
        seg.to_csv(REPORTS_DIR / "resultado_por_ano.csv", index=False)
        logger.success("reports/resultado_por_ano.csv salvo")


def avaliar_cartao(df: pd.DataFrame, pipeline, feat_cols: list[str]) -> None:
    """Avalia modelo de cartão vermelho."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    X = df[feat_cols]
    y = df[TARGET_CARTAO]
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    logger.info(f"[cartão vermelho] ROC-AUC={roc_auc_score(y, y_proba):.4f} | "
                f"AUC-PR={average_precision_score(y, y_proba):.4f}")
    logger.info("\n" + classification_report(y, y_pred, target_names=["sem_vermelho", "com_vermelho"]))


def gerar_relatorio_drift(df_ref: pd.DataFrame, df_atual: pd.DataFrame, feat_cols: list[str]) -> None:
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        logger.warning("Evidently não instalado. Execute: pip install evidently")
        return

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_ref[feat_cols], current_data=df_atual[feat_cols])
    out = REPORTS_DIR / "drift_report.html"
    report.save_html(str(out))
    logger.success(f"Relatório de drift salvo em {out}")


if __name__ == "__main__":
    raw_files = sorted(Path("data/raw").glob("brasileirao_*.parquet"))
    if not raw_files:
        raise FileNotFoundError("Execute 'python -m src.data_ingestion' primeiro.")

    df = pd.read_parquet(raw_files[-1])
    feat_cols = get_feature_cols(df)
    artifact = joblib.load("models/model_final.joblib")

    logger.info("=== Avaliação: Resultado ===")
    avaliar_resultado(df, artifact["pipeline_resultado"], artifact["label_encoder_resultado"], feat_cols)

    logger.info("=== Avaliação: Cartão Vermelho ===")
    avaliar_cartao(df, artifact["pipeline_cartao"], feat_cols)

    # Drift: referência = primeiros 70%, atual = últimos 30%
    split = int(len(df) * 0.7)
    gerar_relatorio_drift(df.iloc[:split], df.iloc[split:], feat_cols)
