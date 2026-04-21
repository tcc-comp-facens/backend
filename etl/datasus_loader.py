"""
ETL — Ingestão de dados DataSUS para Sorocaba (IBGE 355220).

Baixa dados via PySUS, salva em data/datasus/ como cache local.
Na próxima execução, lê do cache se já existir.

Uso:
    python -m etl.datasus_loader [year_from] [year_to]

Sem argumentos, auto-detecta anos dos arquivos SIOPS em data/.
"""

import sys
import uuid
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MUNICIPIO_SOROCABA = "355220"
CACHE_DIR = Path(__file__).parent.parent / "data" / "datasus"
# Cache global pré-baixado pelo script download_pysus.py na raiz do projeto
GLOBAL_CACHE_DIR = Path(__file__).parent.parent.parent / "datasus_cache"

_SINAN_MUN_COLS = ["CODMUNRES", "CO_MUN_RES", "MUNIC_RES", "ID_MUNICIP"]
_SIM_MUN_COLS = ["CODMUNRES", "CO_MUN_RES", "MUNIC_RES", "CODMUNOCOR"]
_SIH_MUN_COLS = ["MUNIC_RES", "CO_MUN_RES", "CODMUNRES", "MUNIC_MOV"]
_PNI_MUN_COLS = ["CO_MUN_RES", "CODMUNRES", "MUNIC_RES", "CO_MUNICIPIO_IBGE", "MUNIC"]

# Persistência Neo4j
_MERGE_QUERY = """
MERGE (i:IndicadorDataSUS {sistema: $sistema, tipo: $tipo, ano: $ano})
SET i.id         = COALESCE(i.id, $id),
    i.valor      = $valor,
    i.fonte      = 'datasus',
    i.importedAt = $importedAt
"""


def _find_mun_col(columns: list, candidates: list[str]) -> Optional[str]:
    cols_upper = {c.upper(): c for c in columns}
    for candidate in candidates:
        if candidate.upper() in cols_upper:
            return cols_upper[candidate.upper()]
    return None


def _filter_sorocaba(df: pd.DataFrame, mun_col: str) -> pd.DataFrame:
    col = df[mun_col].astype(str).str.strip().str.lstrip("0")
    target = MUNICIPIO_SOROCABA.lstrip("0")
    return df[col == target]


def _cache_path(sistema: str, tipo: str, year: int) -> Path:
    return CACHE_DIR / f"{sistema}_{tipo}_{year}.parquet"


def _has_cache(sistema: str, tipo: str, year: int) -> bool:
    return _cache_path(sistema, tipo, year).exists()


def _global_cache_path(sistema: str, tipo: str, year: int) -> Path:
    return GLOBAL_CACHE_DIR / f"{sistema}_{tipo}_{year}.parquet"


def _has_global_cache(sistema: str, tipo: str, year: int) -> bool:
    return _global_cache_path(sistema, tipo, year).exists()


def _save_cache(df: pd.DataFrame, sistema: str, tipo: str, year: int) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(sistema, tipo, year)
    df.to_parquet(path, index=False)
    logger.info("Cache salvo: %s (%d linhas)", path.name, len(df))


def _load_cache(sistema: str, tipo: str, year: int) -> pd.DataFrame:
    # Tenta cache local primeiro, depois cache global (pysus/)
    path = _cache_path(sistema, tipo, year)
    if not path.exists():
        path = _global_cache_path(sistema, tipo, year)
    df = pd.read_parquet(path)
    logger.info("Cache carregado: %s (%d linhas)", path.name, len(df))
    return df


def _persist_batch(session, records: list[dict]) -> int:
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
            importedAt=imported_at,
        )
        count += 1
    return count


# ---------------------------------------------------------------------------
# Download + cache por sistema
# ---------------------------------------------------------------------------


def _read_pysus_result(result) -> pd.DataFrame:
    """Lê o resultado do PySUS independente do formato retornado.

    O PySUS pode retornar:
      - Um objeto com .path (ParquetSet)
      - Uma lista de objetos com .path
      - Uma lista de caminhos (strings/Path)
      - Um único caminho (string/Path)
    """
    # Caso 1: objeto com .path (ParquetSet)
    if hasattr(result, "path"):
        return pd.read_parquet(result.path)

    # Caso 2: lista
    if isinstance(result, list):
        dfs = []
        for item in result:
            if hasattr(item, "path"):
                dfs.append(pd.read_parquet(item.path))
            elif isinstance(item, (str, Path)):
                dfs.append(pd.read_parquet(item))
            else:
                # Tentar ler diretamente como parquet
                dfs.append(pd.read_parquet(item))
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    # Caso 3: caminho direto
    if isinstance(result, (str, Path)):
        return pd.read_parquet(result)

    raise TypeError(f"Formato de resultado PySUS não reconhecido: {type(result)}")


def _download_sinan(disease: str, tipo: str, year: int) -> Optional[pd.DataFrame]:
    """Baixa SINAN via PySUS, filtra Sorocaba e cacheia."""
    if _has_cache("sinan", tipo, year) or _has_global_cache("sinan", tipo, year):
        return _load_cache("sinan", tipo, year)

    try:
        from pysus.online_data import SINAN
        logger.info("SINAN %s: baixando ano %d do FTP...", disease, year)
        result = SINAN.download(diseases=disease, years=year)

        df = _read_pysus_result(result)

        mun_col = _find_mun_col(list(df.columns), _SINAN_MUN_COLS)
        if mun_col is None:
            logger.warning("SINAN %s %d: coluna municipio nao encontrada", disease, year)
            return None

        sorocaba = _filter_sorocaba(df, mun_col)
        _save_cache(sorocaba, "sinan", tipo, year)
        return sorocaba

    except Exception as exc:
        logger.warning("SINAN %s %d: falha — %s", disease, year, exc)
        return None


def _download_sim(year: int) -> Optional[pd.DataFrame]:
    """Baixa SIM via PySUS, filtra Sorocaba e cacheia."""
    if _has_cache("sim", "mortalidade", year) or _has_global_cache("sim", "mortalidade", year):
        return _load_cache("sim", "mortalidade", year)

    try:
        from pysus.online_data import SIM
        logger.info("SIM: baixando ano %d do FTP...", year)
        result = SIM.download(groups="CID10", states="SP", years=year)

        df = _read_pysus_result(result)

        mun_col = _find_mun_col(list(df.columns), _SIM_MUN_COLS)
        if mun_col is None:
            logger.warning("SIM %d: coluna municipio nao encontrada", year)
            return None

        sorocaba = _filter_sorocaba(df, mun_col)
        _save_cache(sorocaba, "sim", "mortalidade", year)
        return sorocaba

    except Exception as exc:
        logger.warning("SIM %d: falha — %s", year, exc)
        return None


def _download_sih(year: int) -> Optional[pd.DataFrame]:
    """Baixa SIH via PySUS, filtra Sorocaba e cacheia."""
    if _has_cache("sih", "internacoes", year) or _has_global_cache("sih", "internacoes", year):
        return _load_cache("sih", "internacoes", year)

    try:
        from pysus.online_data import SIH
        logger.info("SIH: baixando ano %d do FTP...", year)
        months = list(range(1, 13))
        result = SIH.download(states="SP", years=year, months=months, groups="RD")

        df = _read_pysus_result(result)

        mun_col = _find_mun_col(list(df.columns), _SIH_MUN_COLS)
        if mun_col is None:
            logger.warning("SIH %d: coluna municipio nao encontrada", year)
            return None

        sorocaba = _filter_sorocaba(df, mun_col)
        _save_cache(sorocaba, "sih", "internacoes", year)
        return sorocaba

    except Exception as exc:
        logger.warning("SIH %d: falha — %s", year, exc)
        return None


def _download_pni(year: int) -> Optional[pd.DataFrame]:
    """Baixa PNI via PySUS, filtra Sorocaba e cacheia."""
    if _has_cache("si_pni", "vacinacao", year) or _has_global_cache("si_pni", "vacinacao", year):
        return _load_cache("si_pni", "vacinacao", year)

    try:
        from pysus.online_data import PNI
        logger.info("SI-PNI: baixando ano %d do FTP...", year)
        result = PNI.download(group="CPNI", states="SP", years=year)

        df = _read_pysus_result(result)

        mun_col = _find_mun_col(list(df.columns), _PNI_MUN_COLS)
        if mun_col is None:
            logger.warning("SI-PNI %d: coluna municipio nao encontrada", year)
            return None

        sorocaba = _filter_sorocaba(df, mun_col)
        _save_cache(sorocaba, "si_pni", "vacinacao", year)
        return sorocaba

    except Exception as exc:
        logger.warning("SI-PNI %d: falha — %s", year, exc)
        return None


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------


def load(neo4j_client, year_from: int = 2019, year_to: int = 2025,
         siops_years: Optional[set[int]] = None) -> dict:
    """Baixa (ou lê do cache) dados DataSUS e persiste no Neo4j.

    Se siops_years for fornecido, só baixa dados novos do FTP para esses anos.
    Dados já em cache são carregados independentemente.
    """
    logger.info("DataSUS ETL: Sorocaba (%s), anos %d-%d.", MUNICIPIO_SOROCABA, year_from, year_to)
    if siops_years:
        logger.info("DataSUS ETL: anos SIOPS disponíveis = %s (só baixa novos para estes)", sorted(siops_years))

    records: list[dict] = []

    for year in range(year_from, year_to + 1):
        # Se não tem planilha SIOPS para este ano E não tem cache, pula
        has_siops = siops_years is None or year in siops_years

        # Dengue
        if has_siops or _has_cache("sinan", "dengue", year) or _has_global_cache("sinan", "dengue", year):
            df = _download_sinan("DENG", "dengue", year) if has_siops else _load_cache("sinan", "dengue", year) if (_has_cache("sinan", "dengue", year) or _has_global_cache("sinan", "dengue", year)) else None
        else:
            df = None
        if df is not None and len(df) > 0:
            records.append({"sistema": "sinan", "tipo": "dengue", "ano": year,
                            "valor": float(len(df)), "fonte": "datasus"})

        # COVID
        if has_siops or _has_cache("sinan", "covid", year) or _has_global_cache("sinan", "covid", year):
            df = _download_sinan("INFL", "covid", year) if has_siops else _load_cache("sinan", "covid", year) if (_has_cache("sinan", "covid", year) or _has_global_cache("sinan", "covid", year)) else None
        else:
            df = None
        if df is not None and len(df) > 0:
            records.append({"sistema": "sinan", "tipo": "covid", "ano": year,
                            "valor": float(len(df)), "fonte": "datasus"})

        # Vacinação
        if has_siops or _has_cache("si_pni", "vacinacao", year) or _has_global_cache("si_pni", "vacinacao", year):
            df = _download_pni(year) if has_siops else _load_cache("si_pni", "vacinacao", year) if (_has_cache("si_pni", "vacinacao", year) or _has_global_cache("si_pni", "vacinacao", year)) else None
        else:
            df = None
        if df is not None and len(df) > 0:
            records.append({"sistema": "si_pni", "tipo": "vacinacao", "ano": year,
                            "valor": float(len(df)), "fonte": "datasus"})

        # Mortalidade
        if has_siops or _has_cache("sim", "mortalidade", year) or _has_global_cache("sim", "mortalidade", year):
            df = _download_sim(year) if has_siops else _load_cache("sim", "mortalidade", year) if (_has_cache("sim", "mortalidade", year) or _has_global_cache("sim", "mortalidade", year)) else None
        else:
            df = None
        if df is not None and len(df) > 0:
            records.append({"sistema": "sim", "tipo": "mortalidade", "ano": year,
                            "valor": float(len(df)), "fonte": "datasus"})

        # Internações
        if has_siops or _has_cache("sih", "internacoes", year) or _has_global_cache("sih", "internacoes", year):
            df = _download_sih(year) if has_siops else _load_cache("sih", "internacoes", year) if (_has_cache("sih", "internacoes", year) or _has_global_cache("sih", "internacoes", year)) else None
        else:
            df = None
        if df is not None and len(df) > 0:
            records.append({"sistema": "sih", "tipo": "internacoes", "ano": year,
                            "valor": float(len(df)), "fonte": "datasus"})

    if not records:
        logger.warning("DataSUS ETL: nenhum registro obtido.")
        return {"sinan": 0, "si_pni": 0, "sim": 0, "sih": 0}

    counts: dict[str, int] = {}
    with neo4j_client._driver.session() as session:
        _persist_batch(session, records)
    for rec in records:
        counts[rec["sistema"]] = counts.get(rec["sistema"], 0) + 1

    logger.info("DataSUS ETL concluido: %s", counts)
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args = sys.argv[1:]

    from etl.detect_years import detect_siops_years
    siops_years_list = detect_siops_years()
    siops_years_set = set(siops_years_list) if siops_years_list else None

    if len(args) >= 2:
        year_from = int(args[0])
        year_to = int(args[1])
    elif len(args) == 1:
        year_from = int(args[0])
        year_to = year_from
    else:
        if siops_years_list:
            year_from = min(siops_years_list)
            year_to = max(siops_years_list)
            print(f"Anos SIOPS detectados: {siops_years_list} -> buscando {year_from}-{year_to}")
        else:
            year_from = 2019
            year_to = 2025
            print(f"Nenhum SIOPS encontrado, usando padrao {year_from}-{year_to}")

    from db.neo4j_client import Neo4jClient

    with Neo4jClient(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
    ) as client:
        result = load(client, year_from=year_from, year_to=year_to,
                       siops_years=siops_years_set)
        total = sum(result.values())
        print(f"Concluido: {total} registros. Detalhes: {result}")
