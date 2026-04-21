#!/bin/bash
set -e

echo "=== Aguardando Neo4j ficar pronto ==="
MAX_RETRIES=30
RETRY=0
until python -c "
from neo4j import GraphDatabase
import os
d = GraphDatabase.driver(os.environ['NEO4J_URI'], auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD']))
d.verify_connectivity()
d.close()
print('Neo4j conectado.')
" 2>/dev/null; do
    RETRY=$((RETRY + 1))
    if [ $RETRY -ge $MAX_RETRIES ]; then
        echo "Neo4j não respondeu após $MAX_RETRIES tentativas."
        exit 1
    fi
    echo "Neo4j não está pronto (tentativa $RETRY/$MAX_RETRIES)..."
    sleep 2
done

echo ""
echo "=== Executando ETL ==="

# SIOPS — carrega todos os .xls/.xlsx da pasta data/
for f in data/*.xls data/*.xlsx; do
    if [ -f "$f" ]; then
        echo "Carregando SIOPS: $f"
        python -m etl.siops_loader "$f" || echo "AVISO: falha ao carregar $f"
    fi
done

# DataSUS — busca indicadores apenas para os anos com planilhas SIOPS disponíveis
# Dados já em cache de outros anos continuam acessíveis, mas não busca novos
echo "Detectando anos das planilhas SIOPS para limitar busca DataSUS..."
python -m etl.datasus_loader || echo "AVISO: falha no ETL DataSUS"

# Seed de fallback — garante dados mínimos caso ETLs falhem
echo "Executando seed de dados de fallback..."
python -m etl.seed_data || echo "AVISO: falha no seed"

echo ""
echo "=== ETL concluído. Iniciando backend ==="
exec uvicorn main:app --host 0.0.0.0 --port 8000
