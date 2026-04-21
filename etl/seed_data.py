"""
Seed — popula o Neo4j com dados de Sorocaba (2019–2021) como fallback.
"""

import os
import uuid
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DESPESAS = [
    {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2019, "valor": 185420000.0},
    {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2020, "valor": 198750000.0},
    {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2021, "valor": 22552039.69},
    {"subfuncao": 302, "subfuncaoNome": "Assistência Hospitalar", "ano": 2019, "valor": 312800000.0},
    {"subfuncao": 302, "subfuncaoNome": "Assistência Hospitalar", "ano": 2020, "valor": 358200000.0},
    {"subfuncao": 302, "subfuncaoNome": "Assistência Hospitalar", "ano": 2021, "valor": 158643305.49},
    {"subfuncao": 303, "subfuncaoNome": "Suporte Profilático", "ano": 2019, "valor": 42100000.0},
    {"subfuncao": 303, "subfuncaoNome": "Suporte Profilático", "ano": 2020, "valor": 48900000.0},
    {"subfuncao": 303, "subfuncaoNome": "Suporte Profilático", "ano": 2021, "valor": 4061635.28},
    {"subfuncao": 305, "subfuncaoNome": "Vigilância Epidemiológica", "ano": 2019, "valor": 28350000.0},
    {"subfuncao": 305, "subfuncaoNome": "Vigilância Epidemiológica", "ano": 2020, "valor": 45600000.0},
    {"subfuncao": 305, "subfuncaoNome": "Vigilância Epidemiológica", "ano": 2021, "valor": 4886620.43},
]

INDICADORES = [
    {"sistema": "sinan", "tipo": "dengue", "ano": 2019, "valor": 12847.0},
    {"sistema": "sinan", "tipo": "dengue", "ano": 2020, "valor": 5231.0},
    {"sistema": "sinan", "tipo": "dengue", "ano": 2021, "valor": 3412.0},
    {"sistema": "sinan", "tipo": "covid", "ano": 2019, "valor": 0.0},
    {"sistema": "sinan", "tipo": "covid", "ano": 2020, "valor": 28945.0},
    {"sistema": "sinan", "tipo": "covid", "ano": 2021, "valor": 41230.0},
    {"sistema": "si_pni", "tipo": "vacinacao", "ano": 2019, "valor": 485320.0},
    {"sistema": "si_pni", "tipo": "vacinacao", "ano": 2020, "valor": 412100.0},
    {"sistema": "si_pni", "tipo": "vacinacao", "ano": 2021, "valor": 892450.0},
    {"sistema": "sih", "tipo": "internacoes", "ano": 2019, "valor": 45230.0},
    {"sistema": "sih", "tipo": "internacoes", "ano": 2020, "valor": 38750.0},
    {"sistema": "sih", "tipo": "internacoes", "ano": 2021, "valor": 42180.0},
    {"sistema": "sim", "tipo": "mortalidade", "ano": 2019, "valor": 4125.0},
    {"sistema": "sim", "tipo": "mortalidade", "ano": 2020, "valor": 5340.0},
    {"sistema": "sim", "tipo": "mortalidade", "ano": 2021, "valor": 5890.0},
]

_MERGE_DESPESA = """
MERGE (d:DespesaSIOPS {subfuncao: $subfuncao, ano: $ano})
SET d.id            = COALESCE(d.id, $id),
    d.subfuncaoNome = $subfuncaoNome,
    d.valor         = $valor,
    d.fonte         = 'siops',
    d.importedAt    = $importedAt
"""

_MERGE_INDICADOR = """
MERGE (i:IndicadorDataSUS {sistema: $sistema, tipo: $tipo, ano: $ano})
SET i.id         = COALESCE(i.id, $id),
    i.valor      = $valor,
    i.fonte      = 'datasus',
    i.importedAt = $importedAt
"""


def seed(neo4j_client) -> dict:
    imported_at = datetime.now(timezone.utc).isoformat()
    counts = {"despesas": 0, "indicadores": 0}

    with neo4j_client._driver.session() as session:
        for d in DESPESAS:
            session.run(
                _MERGE_DESPESA,
                id=str(uuid.uuid4()),
                subfuncao=d["subfuncao"],
                subfuncaoNome=d["subfuncaoNome"],
                ano=d["ano"],
                valor=d["valor"],
                importedAt=imported_at,
            )
            counts["despesas"] += 1

        for i in INDICADORES:
            session.run(
                _MERGE_INDICADOR,
                id=str(uuid.uuid4()),
                sistema=i["sistema"],
                tipo=i["tipo"],
                ano=i["ano"],
                valor=i["valor"],
                importedAt=imported_at,
            )
            counts["indicadores"] += 1

    logger.info("Seed concluido: %s", counts)
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from db.neo4j_client import Neo4jClient

    with Neo4jClient(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
    ) as client:
        result = seed(client)
        total = result["despesas"] + result["indicadores"]
        print(f"Seed concluido: {total} nos criados/atualizados.")
