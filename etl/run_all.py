"""
ETL completo — carrega SIOPS + DataSUS para os anos dos arquivos em data/.

Uso:
    python -m etl.run_all

1. Detecta os anos dos arquivos SIOPS na pasta data/
2. Carrega cada arquivo SIOPS no Neo4j
3. Baixa dados do DataSUS (PySUS) para o mesmo período
"""

import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    from db.neo4j_client import Neo4jClient
    from etl.siops_loader import load as load_siops
    from etl.datasus_loader import load as load_datasus
    from etl.detect_years import detect_siops_years

    with Neo4jClient(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
    ) as client:

        # 1. Carregar todos os arquivos SIOPS
        siops_files = [f for f in DATA_DIR.iterdir()
                       if f.suffix.lower() in (".xls", ".xlsx")]

        if not siops_files:
            print("Nenhum arquivo SIOPS encontrado em data/")
            sys.exit(1)

        total_siops = 0
        for f in siops_files:
            print(f"\n{'='*60}")
            print(f"Carregando SIOPS: {f.name}")
            print(f"{'='*60}")
            count = load_siops(str(f), client)
            total_siops += count

        print(f"\nSIOPS total: {total_siops} nós persistidos.")

        # 2. Detectar anos e baixar DataSUS
        years = detect_siops_years()
        if not years:
            print("Não foi possível detectar anos dos arquivos SIOPS.")
            return

        year_from = min(years)
        year_to = max(years)
        print(f"\n{'='*60}")
        print(f"Baixando DataSUS para {year_from}–{year_to}")
        print(f"{'='*60}")

        result = load_datasus(client, year_from=year_from, year_to=year_to)
        total_datasus = sum(result.values())

        print(f"\nDataSUS total: {total_datasus} nós persistidos.")
        print(f"Detalhes: {result}")

        print(f"\n{'='*60}")
        print(f"ETL completo: {total_siops + total_datasus} nós no total.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
