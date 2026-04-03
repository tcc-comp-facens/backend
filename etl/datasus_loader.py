"""
ETL — Ingestão de dados DataSUS para o município de Sorocaba (IBGE 355220).

Baixa dados do SINAN (dengue, COVID), SI-PNI (vacinação), SIM (mortalidade)
e SIH (internações) via PySUS, filtra o município 355220, normaliza e persiste
nós `IndicadorDataSUS` no Neo4j com deduplicação via MERGE por (sistema, tipo, ano).

Uso via linha de comando:
    python -m etl.datasus_loader [year_from] [year_to]

Variáveis de ambiente necessárias (ou via .env):
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

import sys
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MUNICIPIO_SOROCABA = "355220"

# Mapeamento sistema -> (tipo, colunas candidatas de município)
_SINAN_MUN_COLS = ["CODMUNRES", "CO_MUN_RES", "MUNIC_RES", "ID_MUNICIP"]
_SIM_MUN_COLS = ["CODMUNRES", "CO_MUN_RES", "MUNIC_RES", "CODMUNOCOR"]
_SIH_MUN_COLS = ["MUNIC_RES", "CO_MUN_RES", "CODMUNRES", "MUNIC_MOV"]
_PNI_MUN_COLS = ["CO_MUN_RES", "CODMUNRES", "MUNIC_RES", "CO_MUNICIPIO_IBGE"]

# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

_MERGE_QUERY = """
MERGE (i:IndicadorDataSUS {sistema: $sistema, tipo: $tipo, ano: $ano})
SET i.id         = COALESCE(i.id, $id),
    i.valor      = $valor,
    i.fonte      = $fonte,
    i.importedAt = $importedAt
"""


def _persist_batch(session, records: list[dict]) -> int:
    """Persiste uma lista de registros normalizados. Retorna o número de nós afetados."""
    imported_at = datetime.now(timezone.utc).isoformat()
    count = 0
    for rec in records:
        session.run(
            _MERGE_QUERY,
            id=str(uuid.uuid4()),
            sistema=rec["sistema"],
            tipo=rec["tipo"],
            ano=rec["ano"],
            valor=rec["valor"],
            fonte=rec["fonte"],
            importedAt=imported_at,
        )
        count += 1
    return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_mun_col(df_columns: list, candidates: list[str]) -> Optional[str]:
    """Retorna a primeira coluna candidata presente no DataFrame."""
    cols_upper = {c.upper(): c for c in df_columns}
    for candidate in candidates:
        if candidate.upper() in cols_upper:
            return cols_upper[candidate.upper()]
    return None


def _filter_sorocaba(df, mun_col: str):
    """Filtra linhas do município de Sorocaba (355220) de forma tolerante a tipos."""
    col = df[mun_col].astype(str).str.strip().str.lstrip("0")
    target = MUNICIPIO_SOROCABA.lstrip("0")
    return df[col == target]


def _aggregate_by_year(df, year_col: str) -> dict[int, float]:
    """Agrega contagem de registros por ano."""
    result: dict[int, float] = {}
    for val in df[year_col]:
        try:
            year = int(str(val).strip()[:4])
            result[year] = result.get(year, 0.0) + 1.0
        except (ValueError, TypeError):
            continue
    return result


# ---------------------------------------------------------------------------
# Loaders por sistema
# ---------------------------------------------------------------------------


def _load_sinan_disease(disease_code: str, tipo: str, year_from: int, year_to: int) -> list[dict]:
    """
    Baixa dados do SINAN para uma doença específica, filtra Sorocaba e
    agrega por ano. Retorna lista de registros normalizados.
    """
    records: list[dict] = []
    try:
        from pysus.online_data import SINAN  # noqa: PLC0415
    except ImportError:
        logger.warning("PySUS não instalado — pulando SINAN %s.", disease_code)
        return records

    for year in range(year_from, year_to + 1):
        try:
            logger.info("SINAN %s: baixando ano %d…", disease_code, year)
            df = SINAN.download(disease=disease_code, year=year)
            if df is None or df.empty:
                logger.warning("SINAN %s %d: sem dados.", disease_code, year)
                continue

            mun_col = _find_mun_col(list(df.columns), _SINAN_MUN_COLS)
            if mun_col is None:
                logger.warning(
                    "SINAN %s %d: coluna de município não encontrada. Colunas: %s",
                    disease_code, year, list(df.columns),
                )
                continue

            sorocaba = _filter_sorocaba(df, mun_col)
            count = float(len(sorocaba))
            if count == 0:
                logger.info("SINAN %s %d: nenhum registro para Sorocaba.", disease_code, year)
                continue

            records.append({
                "sistema": "sinan",
                "tipo": tipo,
                "ano": year,
                "valor": count,
                "fonte": "datasus",
            })
            logger.info("SINAN %s %d: %d notificações em Sorocaba.", disease_code, year, int(count))

        except Exception as exc:  # noqa: BLE001
            logger.warning("SINAN %s %d: falha no download — %s", disease_code, year, exc)

    return records


def _load_pni(year_from: int, year_to: int) -> list[dict]:
    """
    Baixa dados do SI-PNI (vacinação), filtra Sorocaba e agrega por ano.
    """
    records: list[dict] = []
    try:
        from pysus.online_data import PNI  # noqa: PLC0415
    except ImportError:
        logger.warning("PySUS não instalado — pulando SI-PNI.")
        return records

    for year in range(year_from, year_to + 1):
        try:
            logger.info("SI-PNI: baixando ano %d…", year)
            df = PNI.download(year=year)
            if df is None or df.empty:
                logger.warning("SI-PNI %d: sem dados.", year)
                continue

            mun_col = _find_mun_col(list(df.columns), _PNI_MUN_COLS)
            if mun_col is None:
                logger.warning(
                    "SI-PNI %d: coluna de município não encontrada. Colunas: %s",
                    year, list(df.columns),
                )
                continue

            sorocaba = _filter_sorocaba(df, mun_col)
            count = float(len(sorocaba))
            if count == 0:
                logger.info("SI-PNI %d: nenhum registro para Sorocaba.", year)
                continue

            records.append({
                "sistema": "si_pni",
                "tipo": "vacinacao",
                "ano": year,
                "valor": count,
                "fonte": "datasus",
            })
            logger.info("SI-PNI %d: %d doses registradas em Sorocaba.", year, int(count))

        except Exception as exc:  # noqa: BLE001
            logger.warning("SI-PNI %d: falha no download — %s", year, exc)

    return records


def _load_sim(year_from: int, year_to: int) -> list[dict]:
    """
    Baixa dados do SIM (mortalidade), filtra Sorocaba e agrega por ano.
    """
    records: list[dict] = []
    try:
        from pysus.online_data import SIM  # noqa: PLC0415
    except ImportError:
        logger.warning("PySUS não instalado — pulando SIM.")
        return records

    for year in range(year_from, year_to + 1):
        try:
            logger.info("SIM: baixando ano %d…", year)
            df = SIM.download(year=year, state="SP")
            if df is None or df.empty:
                logger.warning("SIM %d: sem dados.", year)
                continue

            mun_col = _find_mun_col(list(df.columns), _SIM_MUN_COLS)
            if mun_col is None:
                logger.warning(
                    "SIM %d: coluna de município não encontrada. Colunas: %s",
                    year, list(df.columns),
                )
                continue

            sorocaba = _filter_sorocaba(df, mun_col)
            count = float(len(sorocaba))
            if count == 0:
                logger.info("SIM %d: nenhum registro para Sorocaba.", year)
                continue

            records.append({
                "sistema": "sim",
                "tipo": "mortalidade",
                "ano": year,
                "valor": count,
                "fonte": "datasus",
            })
            logger.info("SIM %d: %d óbitos em Sorocaba.", year, int(count))

        except Exception as exc:  # noqa: BLE001
            logger.warning("SIM %d: falha no download — %s", year, exc)

    return records


def _load_sih(year_from: int, year_to: int) -> list[dict]:
    """
    Baixa dados do SIH (internações), filtra Sorocaba e agrega por ano.
    """
    records: list[dict] = []
    try:
        from pysus.online_data import SIH  # noqa: PLC0415
    except ImportError:
        logger.warning("PySUS não instalado — pulando SIH.")
        return records

    for year in range(year_from, year_to + 1):
        try:
            logger.info("SIH: baixando ano %d…", year)
            df = SIH.download(year=year, state="SP")
            if df is None or df.empty:
                logger.warning("SIH %d: sem dados.", year)
                continue

            mun_col = _find_mun_col(list(df.columns), _SIH_MUN_COLS)
            if mun_col is None:
                logger.warning(
                    "SIH %d: coluna de município não encontrada. Colunas: %s",
                    year, list(df.columns),
                )
                continue

            sorocaba = _filter_sorocaba(df, mun_col)
            count = float(len(sorocaba))
            if count == 0:
                logger.info("SIH %d: nenhum registro para Sorocaba.", year)
                continue

            records.append({
                "sistema": "sih",
                "tipo": "internacoes",
                "ano": year,
                "valor": count,
                "fonte": "datasus",
            })
            logger.info("SIH %d: %d internações em Sorocaba.", year, int(count))

        except Exception as exc:  # noqa: BLE001
            logger.warning("SIH %d: falha no download — %s", year, exc)

    return records


# ---------------------------------------------------------------------------
# Ponto de entrada público
# ---------------------------------------------------------------------------


def load(neo4j_client, year_from: int = 2018, year_to: int = 2023) -> dict:
    """
    Baixa dados de todos os sistemas DataSUS para Sorocaba (355220),
    normaliza e persiste como nós `IndicadorDataSUS` no Neo4j.

    Usa MERGE por (sistema, tipo, ano) para deduplicação — reimportar
    atualiza os nós existentes em vez de criar duplicatas.

    Retorna dicionário com contagem de nós persistidos por sistema.

    Requisitos: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7
    """
    logger.info(
        "DataSUS ETL: iniciando ingestão para Sorocaba (%s), anos %d–%d.",
        MUNICIPIO_SOROCABA, year_from, year_to,
    )

    all_records: list[dict] = []

    # SINAN — dengue (Req 2.1)
    all_records.extend(_load_sinan_disease("DENG", "dengue", year_from, year_to))

    # SINAN — COVID (Req 2.1)
    all_records.extend(_load_sinan_disease("COVID19", "covid", year_from, year_to))

    # SI-PNI — vacinação (Req 2.2)
    all_records.extend(_load_pni(year_from, year_to))

    # SIM — mortalidade (Req 2.3)
    all_records.extend(_load_sim(year_from, year_to))

    # SIH — internações (Req 2.4)
    all_records.extend(_load_sih(year_from, year_to))

    if not all_records:
        logger.warning("DataSUS ETL: nenhum registro obtido para persistir.")
        return {"sinan": 0, "si_pni": 0, "sim": 0, "sih": 0}

    # Persistência (Req 2.5, 2.6, 2.7)
    counts: dict[str, int] = {"sinan": 0, "si_pni": 0, "sim": 0, "sih": 0}
    with neo4j_client._driver.session() as session:
        _persist_batch(session, all_records)
    for rec in all_records:
        counts[rec["sistema"]] = counts.get(rec["sistema"], 0) + 1

    logger.info(
        "DataSUS ETL: persistência concluída — %s",
        ", ".join(f"{k}: {v}" for k, v in counts.items()),
    )
    return counts


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args = sys.argv[1:]
    year_from = int(args[0]) if len(args) >= 1 else 2018
    year_to = int(args[1]) if len(args) >= 2 else 2023

    from db.neo4j_client import Neo4jClient  # noqa: E402

    with Neo4jClient(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
    ) as client:
        result = load(client, year_from=year_from, year_to=year_to)
        total = sum(result.values())
        print(f"Importação concluída: {total} nós persistidos/atualizados. Detalhes: {result}")
