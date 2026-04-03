"""
ETL — Ingestão de dados SIOPS para o município de Sorocaba (IBGE 355220).

Lê um CSV exportado do portal SIOPS (siops.saude.gov.br), filtra as linhas
do município 355220, normaliza os campos e persiste nós `DespesaSIOPS` no
Neo4j com deduplicação via MERGE por (subfuncao, ano).

Uso via linha de comando:
    python -m etl.siops_loader <caminho_do_csv>

Variáveis de ambiente necessárias (ou via .env):
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
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

SUBFUNCOES_VALIDAS = {301, 302, 303, 305}

SUBFUNCAO_NOME = {
    301: "Atenção Básica",
    302: "Assistência Hospitalar",
    303: "Suporte Profilático",
    305: "Vigilância Epidemiológica",
}

# Variantes de nome de coluna aceitas (case-insensitive após strip)
_COL_MUNICIPIO = {"co_municipio", "municipio", "cd_municipio", "ibge", "co_ibge"}
_COL_SUBFUNCAO = {"co_subfuncao", "subfuncao", "cd_subfuncao", "subfunção"}
_COL_VALOR = {"vl_despesa", "valor", "vl_total", "despesa"}
_COL_ANO = {"aa_exercicio", "ano", "exercicio", "aa_ano"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_col(headers: list[str], candidates: set[str]) -> Optional[str]:
    """Retorna o primeiro header que bate com um dos candidatos (case-insensitive)."""
    normalized = {h.strip().lower(): h for h in headers}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _detect_columns(headers: list[str]) -> dict[str, str]:
    """
    Detecta os nomes reais das colunas no CSV e retorna um mapeamento
    lógico -> nome_real.

    Lança ValueError se alguma coluna obrigatória não for encontrada.
    """
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
            raise ValueError(
                f"Coluna '{logical}' não encontrada. "
                f"Cabeçalhos disponíveis: {headers}. "
                f"Variantes aceitas: {candidates}"
            )
        mapping[logical] = col
    return mapping


def _parse_int(value: str) -> Optional[int]:
    try:
        return int(str(value).strip().replace(".", "").replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _parse_float(value: str) -> Optional[float]:
    try:
        # Suporta separador decimal vírgula (padrão BR) ou ponto
        cleaned = str(value).strip().replace(".", "").replace(",", ".")
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Normalização
# ---------------------------------------------------------------------------


def normalize_row(row: dict, col_map: dict[str, str]) -> Optional[dict]:
    """
    Normaliza uma linha do CSV para o esquema DespesaSIOPS.

    Retorna None se a linha deve ser descartada (município diferente,
    subfunção fora do escopo ou valores inválidos).
    """
    municipio = _parse_int(row.get(col_map["municipio"], ""))
    if municipio != MUNICIPIO_SOROCABA:
        return None

    subfuncao = _parse_int(row.get(col_map["subfuncao"], ""))
    if subfuncao not in SUBFUNCOES_VALIDAS:
        return None

    ano = _parse_int(row.get(col_map["ano"], ""))
    if ano is None:
        return None

    valor = _parse_float(row.get(col_map["valor"], ""))
    if valor is None:
        return None

    return {
        "subfuncao": subfuncao,
        "subfuncaoNome": SUBFUNCAO_NOME[subfuncao],
        "ano": ano,
        "valor": valor,
        "fonte": "siops",
    }


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

_MERGE_QUERY = """
MERGE (d:DespesaSIOPS {subfuncao: $subfuncao, ano: $ano})
SET d.id           = COALESCE(d.id, $id),
    d.subfuncaoNome = $subfuncaoNome,
    d.valor         = $valor,
    d.fonte         = $fonte,
    d.importedAt    = $importedAt
"""


def _persist_batch(session, records: list[dict]) -> int:
    """Persiste uma lista de registros normalizados. Retorna o número de nós afetados."""
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
            fonte=rec["fonte"],
            importedAt=imported_at,
        )
        count += 1
    return count


# ---------------------------------------------------------------------------
# Ponto de entrada público
# ---------------------------------------------------------------------------


def load(csv_path: str, neo4j_client) -> int:
    """
    Lê o CSV do SIOPS em `csv_path`, filtra Sorocaba (355220), normaliza
    e persiste como nós `DespesaSIOPS` no Neo4j.

    Usa MERGE por (subfuncao, ano) para deduplicação — reimportar o mesmo
    arquivo atualiza os nós existentes em vez de criar duplicatas.

    Retorna o número de nós persistidos/atualizados.

    Requisitos: 1.1, 1.2, 1.3, 1.4
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")

    records: list[dict] = []
    col_map: Optional[dict] = None

    # Tenta detectar o encoding (UTF-8 com BOM é comum em exports do governo)
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            with open(path, newline="", encoding=encoding) as fh:
                reader = csv.DictReader(fh, delimiter=";")
                if reader.fieldnames is None:
                    # Tenta com vírgula como delimitador
                    fh.seek(0)
                    reader = csv.DictReader(fh, delimiter=",")

                headers = list(reader.fieldnames or [])
                col_map = _detect_columns(headers)

                for row in reader:
                    normalized = normalize_row(row, col_map)
                    if normalized is not None:
                        records.append(normalized)
            break  # leitura bem-sucedida
        except (UnicodeDecodeError, ValueError):
            records = []
            col_map = None
            continue

    if col_map is None:
        raise ValueError(
            f"Não foi possível ler o CSV '{csv_path}'. "
            "Verifique o encoding e o formato do arquivo."
        )

    logger.info(
        "SIOPS: %d registros válidos encontrados para município %d",
        len(records),
        MUNICIPIO_SOROCABA,
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
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("Uso: python -m etl.siops_loader <caminho_do_csv>", file=sys.stderr)
        sys.exit(1)

    csv_file = sys.argv[1]

    # Importação tardia para não exigir neo4j instalado em testes unitários
    from db.neo4j_client import Neo4jClient  # noqa: E402

    with Neo4jClient(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
    ) as client:
        total = load(csv_file, client)
        print(f"Importação concluída: {total} nós persistidos/atualizados.")
