"""
Ingestão do Campeonato Brasileiro de Futebol via Kaggle.

Fonte: adaoduque/campeonato-brasileiro-de-futebol
Arquivos utilizados:
  - campeonato-brasileiro-full.csv       — dados gerais das partidas (9.165 linhas)
  - campeonato-brasileiro-estatisticas-full.csv — stats por clube/partida (18.330 linhas)
  - campeonato-brasileiro-cartoes.csv    — cartões individuais (20.953 linhas)

SAÍDA
─────
Um DataFrame com uma linha por partida contendo:
  - Identificadores: partida_id, rodata, data, mandante, visitante
  - Features rolantes (últimas 5 partidas de cada time):
      mandante_* e visitante_* para chutes, chutes_no_alvo, posse, passes,
      precisao_passes, faltas, cartao_amarelo, impedimentos, escanteios,
      gols_marcados, gols_sofridos
  - Target 1: resultado — 'mandante' | 'empate' | 'visitante'
  - Target 2: tem_cartao_vermelho — 0 | 1 (partida teve ≥1 cartão vermelho)

ANTI-LEAKAGE
─────────────
As features rolantes são computadas ANTES de cada partida (shift(1) + rolling(5)).
Estatísticas da partida atual NUNCA entram como features.
Targets são derivados exclusivamente de colunas pós-jogo (vencedor, cartao_vermelho).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

try:
    import kagglehub
except ImportError:
    raise ImportError("Instale kagglehub: pip install 'kagglehub[pandas-datasets]'")

DATASET_ID = "adaoduque/campeonato-brasileiro-de-futebol"
ROLLING_WINDOW = 5  # últimas N partidas para média rolante
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

STATS_COLS = [
    "chutes", "chutes_no_alvo", "posse_de_bola", "passes",
    "precisao_passes", "faltas", "cartao_amarelo", "impedimentos", "escanteios",
]


def _load_csvs(local_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = {f: os.path.join(local_path, f) for f in os.listdir(local_path) if f.endswith(".csv")}
    logger.info(f"Arquivos encontrados: {list(files.keys())}")

    df_full = pd.read_csv(files["campeonato-brasileiro-full.csv"], low_memory=False)
    df_stats = pd.read_csv(files["campeonato-brasileiro-estatisticas-full.csv"], low_memory=False)
    df_cartoes = pd.read_csv(files["campeonato-brasileiro-cartoes.csv"], low_memory=False)

    logger.info(f"full: {df_full.shape} | stats: {df_stats.shape} | cartoes: {df_cartoes.shape}")
    return df_full, df_stats, df_cartoes


def _preparar_full(df_full: pd.DataFrame) -> pd.DataFrame:
    """Limpa e ordena o dataframe de partidas."""
    df = df_full.rename(columns={"ID": "partida_id"}).copy()

    # Parsear data — formato "29/03/2003"
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="coerce")
    df = df.sort_values("data").reset_index(drop=True)

    # Target 1: resultado
    def _resultado(row: pd.Series) -> str:
        if row["vencedor"] == row["mandante"]:
            return "mandante"
        elif row["vencedor"] == row["visitante"]:
            return "visitante"
        else:
            return "empate"

    df["resultado"] = df.apply(_resultado, axis=1)

    return df[["partida_id", "rodata", "data", "mandante", "visitante",
               "mandante_Estado", "visitante_Estado", "resultado"]]


def _preparar_stats(df_stats: pd.DataFrame) -> pd.DataFrame:
    """Converte estatísticas para numérico. posse_de_bola e precisao_passes vêm como '45%'."""
    df = df_stats.copy()
    for col in STATS_COLS:
        if df[col].dtype == object:
            # Remove símbolo de percentual antes de converter
            df[col] = df[col].astype(str).str.replace("%", "", regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _adicionar_gols(df_stats: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
    """Adiciona gols marcados e sofridos por clube por partida."""
    df_full_gols = df_full.rename(columns={"ID": "partida_id"})[
        ["partida_id", "mandante", "visitante", "mandante_Placar", "visitante_Placar"]
    ].copy()
    df_full_gols["mandante_Placar"] = pd.to_numeric(df_full_gols["mandante_Placar"], errors="coerce")
    df_full_gols["visitante_Placar"] = pd.to_numeric(df_full_gols["visitante_Placar"], errors="coerce")

    # Melt para ter (partida_id, clube, gols_marcados, gols_sofridos)
    mand = df_full_gols[["partida_id", "mandante", "mandante_Placar", "visitante_Placar"]].copy()
    mand.columns = ["partida_id", "clube", "gols_marcados", "gols_sofridos"]
    vis = df_full_gols[["partida_id", "visitante", "visitante_Placar", "mandante_Placar"]].copy()
    vis.columns = ["partida_id", "clube", "gols_marcados", "gols_sofridos"]

    df_gols = pd.concat([mand, vis], ignore_index=True)
    return df_stats.merge(df_gols, on=["partida_id", "clube"], how="left")


def _target_cartao_vermelho(df_stats: pd.DataFrame) -> pd.Series:
    """Retorna Series partida_id → tem_cartao_vermelho (0/1)."""
    cv_por_partida = (
        df_stats.groupby("partida_id")["cartao_vermelho"]
        .sum()
        .reset_index()
        .rename(columns={"cartao_vermelho": "tem_cartao_vermelho"})
    )
    cv_por_partida["tem_cartao_vermelho"] = (cv_por_partida["tem_cartao_vermelho"] > 0).astype(int)
    return cv_por_partida


def _rolling_por_clube(
    df_full_clean: pd.DataFrame,
    df_stats_com_gols: pd.DataFrame,
) -> pd.DataFrame:
    """
    Para cada partida, computa médias rolantes das últimas ROLLING_WINDOW partidas
    de cada clube ANTES do jogo atual (shift+rolling).

    Retorna DataFrame com exatamente uma linha por partida e colunas mandante_* / visitante_*.
    """
    logger.info(f"Computando features rolantes (janela = {ROLLING_WINDOW})...")

    feature_cols = STATS_COLS + ["gols_marcados", "gols_sofridos"]

    df_stats_ord = df_stats_com_gols.merge(
        df_full_clean[["partida_id", "data"]], on="partida_id", how="left"
    ).sort_values("data")

    roll_frames = []
    for clube, grp in df_stats_ord.groupby("clube"):
        df_c = grp.sort_values("data").copy()
        for col in feature_cols:
            df_c[f"roll_{col}"] = (
                df_c[col]
                .shift(1)
                .rolling(window=ROLLING_WINDOW, min_periods=1)
                .mean()
            )
        roll_cols = ["partida_id", "clube"] + [f"roll_{c}" for c in feature_cols]
        roll_frames.append(df_c[roll_cols])

    # Uma linha por (partida_id, clube)
    df_roll = pd.concat(roll_frames, ignore_index=True)

    lados = df_full_clean[["partida_id", "mandante", "visitante"]]

    # Features do mandante
    df_mand = lados.merge(
        df_roll, left_on=["partida_id", "mandante"], right_on=["partida_id", "clube"], how="left"
    ).drop(columns=["clube", "mandante", "visitante"])
    df_mand = df_mand.rename(columns={c: f"mandante_{c[5:]}" for c in df_mand.columns if c.startswith("roll_")})

    # Features do visitante
    df_vis = lados.merge(
        df_roll, left_on=["partida_id", "visitante"], right_on=["partida_id", "clube"], how="left"
    ).drop(columns=["clube", "mandante", "visitante"])
    df_vis = df_vis.rename(columns={c: f"visitante_{c[5:]}" for c in df_vis.columns if c.startswith("roll_")})

    df_features = df_mand.merge(df_vis, on="partida_id", how="inner")
    return df_features


def ingerir(output_path: Path | None = None) -> pd.DataFrame:
    """
    Baixa e prepara o dataset do Campeonato Brasileiro.

    Retorna DataFrame com uma linha por partida, features rolantes e dois targets.
    Salva em data/raw/brasileirao_YYYYMMDD_HHMM.parquet.
    """
    logger.info(f"Carregando dataset: {DATASET_ID}")
    local_path = kagglehub.dataset_download(DATASET_ID)
    logger.info(f"Dataset em cache: {local_path}")

    df_full_raw, df_stats_raw, df_cartoes_raw = _load_csvs(local_path)

    df_full_clean = _preparar_full(df_full_raw)
    logger.info(f"Partidas após limpeza: {len(df_full_clean)} | intervalo: "
                f"{df_full_clean['data'].min().date()} → {df_full_clean['data'].max().date()}")

    df_stats = _preparar_stats(df_stats_raw)
    df_stats = _adicionar_gols(df_stats, df_full_raw)

    # Target 2 — cartão vermelho por partida
    df_cv = _target_cartao_vermelho(df_stats)
    logger.info(
        f"Taxa de cartão vermelho: "
        f"{df_cv['tem_cartao_vermelho'].mean():.1%} das partidas"
    )

    # Features rolantes
    df_features = _rolling_por_clube(df_full_clean, df_stats)

    # Montar dataset final
    df = (
        df_full_clean
        .merge(df_cv, on="partida_id", how="left")
        .merge(df_features, on="partida_id", how="left")
    )
    df["tem_cartao_vermelho"] = df["tem_cartao_vermelho"].fillna(0).astype(int)

    # Remover partidas sem features rolantes (primeiras de cada time)
    feature_cols = [c for c in df.columns if c.startswith(("mandante_", "visitante_"))
                    and c not in ("mandante_Estado", "visitante_Estado")]
    df_com_features = df.dropna(subset=feature_cols[:4], how="all").copy()

    logger.info(f"Dataset final: {len(df_com_features)} partidas × {df_com_features.shape[1]} colunas")
    logger.info(f"Distribuição resultado: {df_com_features['resultado'].value_counts().to_dict()}")
    logger.info(f"Cartão vermelho: {df_com_features['tem_cartao_vermelho'].value_counts().to_dict()}")

    if output_path is None:
        ts = datetime.today().strftime("%Y%m%d_%H%M")
        output_path = RAW_DIR / f"brasileirao_{ts}.parquet"

    df_com_features.to_parquet(output_path, index=False)
    logger.success(f"Salvo em {output_path}")
    return df_com_features


if __name__ == "__main__":
    ingerir()
