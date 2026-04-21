"""
Detecta os anos disponíveis nos arquivos SIOPS da pasta data/.

Lê os metadados de cada .xls/.xlsx na pasta e extrai o campo "Ano:".
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def _extract_year_from_xls(path: Path) -> Optional[int]:
    """Extrai o ano dos metadados de uma planilha SIOPS."""
    try:
        import pandas as pd
        suffix = path.suffix.lower()
        engine = "xlrd" if suffix == ".xls" else "openpyxl"
        df = pd.read_excel(path, sheet_name=0, header=None, engine=engine, nrows=10)

        for row_idx in range(min(10, len(df))):
            for col_idx in range(min(10, df.shape[1])):
                cell = str(df.iloc[row_idx, col_idx]).strip().lower()
                if cell == "ano:":
                    for c in range(col_idx + 1, min(col_idx + 10, df.shape[1])):
                        val = df.iloc[row_idx, c]
                        if str(val) != "nan":
                            return int(float(str(val).strip()))
    except Exception as exc:
        logger.warning("Não foi possível extrair ano de %s: %s", path.name, exc)
    return None


def detect_siops_years() -> list[int]:
    """Retorna lista ordenada de anos encontrados nos arquivos SIOPS em data/."""
    if not DATA_DIR.exists():
        logger.warning("Pasta data/ não encontrada.")
        return []

    years = set()
    for f in DATA_DIR.iterdir():
        if f.suffix.lower() in (".xls", ".xlsx"):
            year = _extract_year_from_xls(f)
            if year:
                years.add(year)
                logger.info("Arquivo %s → ano %d", f.name, year)

    result = sorted(years)
    logger.info("Anos detectados nos arquivos SIOPS: %s", result)
    return result
