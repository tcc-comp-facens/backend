"""
ETL — Ingestão de dados SIOPS para o município de Sorocaba (IBGE 355220).

Lê planilhas detalhadas exportadas do portal SIOPS (formato .xls/.xlsx/.csv),
extrai metadados (ano, IBGE), mapeia os grupos de despesa para subfunções,
agrega valores por subfunção e persiste nós `DespesaSIOPS` no Neo4j.

Uso:
    python -m etl.siops_loader <caminho_do_arquivo>
    python -m etl.siops_loader data/PlanilhaDetalhada.xls

Variáveis de ambiente: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

import csv
import sys
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MUNICIPIO_SOROCABA = 355220

SUBFUNCAO_NOME = {
    301: "Atenção Básica",
    302: "Assistência Hospitalar",
    303: "Suporte Profilático",
    305: "Vigilância Epidemiológica",
}

# Mapeamento dos nomes de grupo da planilha SIOPS → código de subfunção
GRUPO_TO_SUBFUNCAO: dict[str, int] = {
    "ATENÇÃO PRIMÁRIA": 301,
    "ATENCAO PRIMARIA": 301,
    "ATENÇÃO BÁSICA": 301,
    "ATENCAO BASICA": 301,
}

# Continuação do mapeamento grupo → subfunção
_GRUPO_MAP_EXTRA = {
    "ATENÇÃO DE MÉDIA E ALTA COMPLEXIDADE AMBULATORIAL E HOSPITALAR": 302,
    "ATENÇÃO DE MEDIA E ALTA COMPLEXIDADE AMBULATORIAL E HOSPITALAR": 302,
    "ATENÇÃO ESPECIALIZADA": 302,
    "ATENCAO ESPECIALIZADA": 302,
    "MAC": 302,
    "ASSISTÊNCIA FARMACÊUTICA": 303,
    "ASSISTENCIA FARMACEUTICA": 303,
    "SUPORTE PROFILÁTICO": 303,
    "VIGILÂNCIA EM SAÚDE": 305,
    "VIGILANCIA EM SAUDE": 305,
    "VIGILÂNCIA EPIDEMIOLÓGICA": 305,
    "VIGILANCIA EPIDEMIOLOGICA": 305,
}
GRUPO_TO_SUBFUNCAO.update(_GRUPO_MAP_EXTRA)


def _match_grupo(grupo_raw: str) -> Optional[int]:
    """Tenta mapear um nome de grupo para código de subfunção."""
    grupo = grupo_raw.strip().upper()
    # Busca exata
    if grupo in GRUPO_TO_SUBFUNCAO:
        return GRUPO_TO_SUBFUNCAO[grupo]
    # Busca parcial
    for key, code in GRUPO_TO_SUBFUNCAO.items():
        if key in grupo or grupo in key:
            return code
    return None


def _parse_valor_br(value) -> Optional[float]:
    """Converte valor no formato brasileiro (1.234.567,89) para float."""
    try:
        if isinstance(value, (int, float)):
            return float(value)
        cleaned = str(value).strip().replace(".", "").replace(",", ".")
        return float(cleaned)
    except (ValueError, AttributeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

_MERGE_QUERY = """
MERGE (d:DespesaSIOPS {subfuncao: $subfuncao, ano: $ano})
SET d.id            = COALESCE(d.id, $id),
    d.subfuncaoNome = $subfuncaoNome,
    d.valor         = $valor,
    d.fonte         = 'siops',
    d.importedAt    = $importedAt
"""


def _persist_batch(session, records: list[dict]) -> int:
    imported_at = datetime.now(timezone.utc).isoformat()
    count = 0
    for rec in records:
        session.run(
            _MERGE_QUERY,
            id=str(uuid.uuid4()),
            subfuncao=rec["subfuncao"],
            subfuncaoNome=rec["subfuncaoNome"],
            ano=rec["ano"],
            valor=rec["valor"],
            importedAt=imported_at,
        )
        count += 1
    return count


# ---------------------------------------------------------------------------
# Leitor da planilha detalhada SIOPS
# ---------------------------------------------------------------------------


def _read_planilha_detalhada(path: Path) -> list[dict]:
    """Lê a planilha detalhada do SIOPS (.xls ou .xlsx).

    Estrutura esperada:
      - Linhas 0-6: metadados (Município, Ano, IBGE, etc.)
      - Linha 7: cabeçalho
      - Linhas 8+: dados de repasses
      - Última linha: TOTAL GERAL

    Colunas relevantes:
      - Col 8 (Grupo): nome do grupo de despesa → mapeia para subfunção
      - Col 17 (Valor Total): valor em formato BR
      - Metadado Ano (linha 2, col 7): ano do exercício
    """
    import pandas as pd

    suffix = path.suffix.lower()
    engine = "xlrd" if suffix == ".xls" else "openpyxl"

    df = pd.read_excel(path, sheet_name=0, header=None, engine=engine)
    logger.info("Planilha carregada: %d linhas x %d colunas", df.shape[0], df.shape[1])

    # Extrair ano do metadado
    ano = None
    for row_idx in range(min(7, len(df))):
        for col_idx in range(min(10, df.shape[1])):
            cell = str(df.iloc[row_idx, col_idx]).strip().lower()
            if cell == "ano:":
                # O valor do ano está na próxima coluna com conteúdo
                for c in range(col_idx + 1, min(col_idx + 10, df.shape[1])):
                    val = df.iloc[row_idx, c]
                    if str(val) != "nan":
                        try:
                            ano = int(float(str(val).strip()))
                        except (ValueError, TypeError):
                            pass
                        break
                break
        if ano:
            break

    if ano is None:
        logger.warning("Ano não encontrado nos metadados, tentando extrair das datas.")

    logger.info("Ano detectado: %s", ano)

    # Detectar linha de cabeçalho (procura por "Grupo" ou "Bloco")
    header_row = 7  # default
    for i in range(min(15, len(df))):
        row_vals = [str(v).strip().lower() for v in df.iloc[i].values if str(v) != "nan"]
        if any(h in row_vals for h in ["grupo", "bloco", "ação detalhada"]):
            header_row = i
            break

    # Detectar colunas por nome do header
    headers = {str(df.iloc[header_row, j]).strip(): j for j in range(df.shape[1])
               if str(df.iloc[header_row, j]) != "nan"}

    logger.info("Headers detectados: %s", headers)

    # Encontrar coluna do Grupo e Valor Total
    grupo_col = None
    valor_col = None
    for name, idx in headers.items():
        name_lower = name.lower()
        if name_lower == "grupo":
            grupo_col = idx
        elif "valor total" in name_lower:
            valor_col = idx

    if grupo_col is None:
        logger.warning("Coluna 'Grupo' não encontrada, usando col 8 como fallback.")
        grupo_col = 8
    if valor_col is None:
        logger.warning("Coluna 'Valor Total' não encontrada, usando col 17 como fallback.")
        valor_col = 17

    logger.info("Grupo col: %d, Valor col: %d", grupo_col, valor_col)

    # Agregar valores por subfunção
    aggregated: dict[int, float] = {}
    skipped_grupos: set[str] = set()

    for i in range(header_row + 1, len(df)):
        grupo_raw = df.iloc[i, grupo_col]
        if pd.isna(grupo_raw):
            continue

        grupo_str = str(grupo_raw).strip()
        if "TOTAL" in grupo_str.upper():
            continue

        subfuncao = _match_grupo(grupo_str)
        if subfuncao is None:
            skipped_grupos.add(grupo_str)
            continue

        valor = _parse_valor_br(df.iloc[i, valor_col])
        if valor is None or valor <= 0:
            continue

        aggregated[subfuncao] = aggregated.get(subfuncao, 0.0) + valor

    if skipped_grupos:
        logger.info("Grupos não mapeados (ignorados): %s", skipped_grupos)

    # Converter para registros
    records = []
    for subfuncao, valor_total in aggregated.items():
        records.append({
            "subfuncao": subfuncao,
            "subfuncaoNome": SUBFUNCAO_NOME.get(subfuncao, str(subfuncao)),
            "ano": ano or 0,
            "valor": round(valor_total, 2),
            "fonte": "siops",
        })

    return records


# ---------------------------------------------------------------------------
# Leitor CSV legado (mantido para compatibilidade)
# ---------------------------------------------------------------------------

# Variantes de nome de coluna para CSV
_COL_MUNICIPIO = {
    "co_municipio", "municipio", "cd_municipio", "ibge", "co_ibge",
    "id_municipio", "cod_municipio",
}
_COL_SUBFUNCAO = {
    "co_subfuncao", "subfuncao", "cd_subfuncao", "subfunção",
    "id_subfuncao", "cod_subfuncao",
}
_COL_VALOR = {
    "vl_despesa", "valor", "vl_total", "despesa",
    "valor_despesa", "vl_despesa_total",
}
_COL_ANO = {
    "aa_exercicio", "ano", "exercicio", "aa_ano", "ano_exercicio",
}


def _find_col(headers: list[str], candidates: set[str]) -> Optional[str]:
    normalized = {h.strip().lower(): h for h in headers}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _read_csv_legacy(path: Path) -> list[dict]:
    """Lê CSV tabulado do SIOPS (formato antigo com colunas municipio/subfuncao/valor/ano)."""
    records: list[dict] = []

    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            with open(path, newline="", encoding=encoding) as fh:
                reader = csv.DictReader(fh, delimiter=";")
                if reader.fieldnames is None:
                    fh.seek(0)
                    reader = csv.DictReader(fh, delimiter=",")

                headers = list(reader.fieldnames or [])
                mapping = {}
                required = {
                    "municipio": _COL_MUNICIPIO,
                    "subfuncao": _COL_SUBFUNCAO,
                    "valor": _COL_VALOR,
                    "ano": _COL_ANO,
                }
                for logical, candidates in required.items():
                    col = _find_col(headers, candidates)
                    if col is None:
                        raise ValueError(f"Coluna '{logical}' não encontrada.")
                    mapping[logical] = col

                for row in reader:
                    try:
                        mun = int(str(row.get(mapping["municipio"], "")).strip().replace(".", ""))
                    except (ValueError, TypeError):
                        continue
                    if mun != MUNICIPIO_SOROCABA:
                        continue
                    try:
                        sub = int(str(row.get(mapping["subfuncao"], "")).strip())
                    except (ValueError, TypeError):
                        continue
                    if sub not in SUBFUNCAO_NOME:
                        continue
                    valor = _parse_valor_br(row.get(mapping["valor"], ""))
                    try:
                        ano = int(str(row.get(mapping["ano"], "")).strip())
                    except (ValueError, TypeError):
                        continue
                    if valor is not None:
                        records.append({
                            "subfuncao": sub,
                            "subfuncaoNome": SUBFUNCAO_NOME[sub],
                            "ano": ano,
                            "valor": valor,
                            "fonte": "siops",
                        })
            return records
        except (UnicodeDecodeError, ValueError):
            records = []
            continue

    raise ValueError(f"Não foi possível ler o CSV '{path}'.")


# ---------------------------------------------------------------------------
# Ponto de entrada público
# ---------------------------------------------------------------------------


def load(file_path: str, neo4j_client) -> int:
    """Lê arquivo SIOPS, agrega por subfunção e persiste no Neo4j."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        records = _read_planilha_detalhada(path)
    elif suffix == ".csv":
        records = _read_csv_legacy(path)
    else:
        raise ValueError(f"Formato não suportado: {suffix}. Use .csv, .xls ou .xlsx")

    logger.info("SIOPS: %d registros para persistir.", len(records))

    for rec in records:
        logger.info(
            "  Subfuncao %d (%s) - Ano %d - R$ %.2f",
            rec["subfuncao"], rec["subfuncaoNome"], rec["ano"], rec["valor"],
        )

    if not records:
        logger.warning("Nenhum registro para persistir.")
        return 0

    with neo4j_client._driver.session() as session:
        count = _persist_batch(session, records)

    logger.info("SIOPS: %d nós DespesaSIOPS persistidos/atualizados.", count)
    return count


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("Uso: python -m etl.siops_loader <arquivo>", file=sys.stderr)
        print("  Formatos: .csv, .xls, .xlsx", file=sys.stderr)
        print("  Exemplo: python -m etl.siops_loader data/PlanilhaDetalhada.xls", file=sys.stderr)
        sys.exit(1)

    from db.neo4j_client import Neo4jClient

    with Neo4jClient(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
    ) as client:
        total = load(sys.argv[1], client)
        print(f"Importação concluída: {total} nós persistidos/atualizados.")
