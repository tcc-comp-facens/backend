# multiagent-backend

Backend Python + FastAPI para o sistema de comparação de arquiteturas multiagente BDI aplicado à análise de gastos públicos em saúde de Sorocaba-SP.

## Pré-requisitos

- Python 3.10+
- Neo4j 5.x rodando localmente (ou via Docker)
- Chave de API para o LLM (ex: OpenAI)

## Instalação

```bash
# 1. Clone o repositório e entre no diretório
cd multiagent-backend

# 2. Crie e ative um ambiente virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
cp .env.example .env
# Edite o arquivo .env com suas credenciais
```

## Configuração

Edite o arquivo `.env` com os valores corretos:

| Variável | Descrição | Exemplo |
|---|---|---|
| `NEO4J_URI` | URI de conexão com o Neo4j | `bolt://localhost:7687` |
| `NEO4J_USER` | Usuário do Neo4j | `neo4j` |
| `NEO4J_PASSWORD` | Senha do Neo4j | `sua_senha` |
| `LLM_API_KEY` | Chave de API do LLM | `sk-...` |

## Pipeline ETL (pré-carga de dados)

Antes de executar o backend, carregue os dados no Neo4j:

```bash
# Ingestão de dados financeiros do SIOPS
python -m etl.siops_loader

# Ingestão de indicadores de saúde do DataSUS
python -m etl.datasus_loader
```

## Execução com Docker

```bash
# Subir todos os serviços (Neo4j + backend)
docker compose up -d

# Subir apenas o Neo4j (útil para desenvolvimento local)
docker compose up -d neo4j

# Parar todos os serviços
docker compose down
```

## Execução

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

O backend estará disponível em `http://localhost:8000`.

### Endpoints disponíveis

| Método | Rota | Descrição |
|---|---|---|
| `POST` | `/api/analysis` | Inicia uma análise comparativa |
| `GET` | `/api/analysis/{id}` | Consulta resultado de uma análise |
| `GET` | `/api/benchmarks` | Métricas comparativas de todas as análises |
| `WS` | `/ws/{analysisId}` | Stream de eventos em tempo real |

Documentação interativa (Swagger): `http://localhost:8000/docs`

## Testes

```bash
# Executar todos os testes
pytest tests/

# Com cobertura
pytest tests/ --cov=. --cov-report=term-missing

# Apenas testes de propriedade (Hypothesis)
pytest tests/ -k "property"
```

## Estrutura do projeto

```
multiagent-backend/
├── main.py                    # FastAPI app — endpoints REST e WebSocket
├── metrics.py                 # MetricsCollector (psutil)
├── requirements.txt
├── .env.example
├── agents/
│   ├── base.py                # Classe AgenteBDI (compartilhada)
│   ├── star/
│   │   ├── orchestrator.py    # OrquestradorEstrela
│   │   ├── consultant.py      # AgenteConsultorStar
│   │   └── analyzer.py        # AgenteAnalisadorStar
│   └── hierarchical/
│       ├── coordinator.py     # CoordenadorGeral
│       ├── supervisors.py     # SupervisorConsulta, SupervisorAnalise
│       ├── consultant.py      # AgenteConsultorHier
│       └── analyzer.py        # AgenteAnalisadorHier
├── db/
│   └── neo4j_client.py        # Driver Neo4j + queries Cypher
├── etl/
│   ├── siops_loader.py        # Ingestão SIOPS → Neo4j
│   └── datasus_loader.py      # Ingestão DataSUS via PySUS → Neo4j
└── tests/                     # pytest + Hypothesis
```
