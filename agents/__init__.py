"""Pacote de agentes multiagente — exporta todos os agentes especializados.

Organização:
- agents.base: Classe base AgenteBDI
- agents.domain: Agentes de domínio (Req 14.1)
- agents.analytical: Agentes analíticos (Req 14.2)
- agents.context: Agente de contexto (Req 14.3)
- agents.star: Topologia estrela (Req 14.4)
- agents.hierarchical: Topologia hierárquica (Req 14.4)
"""

# Base
from agents.base import AgenteBDI, IntentionFailure

# Domain agents (Req 14.1)
from agents.domain import (
    AgenteVigilanciaEpidemiologica,
    AgenteSaudeHospitalar,
    AgenteAtencaoPrimaria,
    AgenteMortalidade,
)

# Analytical agents (Req 14.2)
from agents.analytical import (
    AgenteCorrelacao,
    AgenteAnomalias,
    AgenteSintetizador,
)

# Context agent (Req 14.3)
from agents.context import AgenteContextoOrcamentario

# Star topology (Req 14.4)
from agents.star import OrquestradorEstrela

# Hierarchical topology (Req 14.4)
from agents.hierarchical import CoordenadorGeral

__all__ = [
    # Base
    "AgenteBDI",
    "IntentionFailure",
    # Domain
    "AgenteVigilanciaEpidemiologica",
    "AgenteSaudeHospitalar",
    "AgenteAtencaoPrimaria",
    "AgenteMortalidade",
    # Analytical
    "AgenteCorrelacao",
    "AgenteAnomalias",
    "AgenteSintetizador",
    # Context
    "AgenteContextoOrcamentario",
    # Star
    "OrquestradorEstrela",
    # Hierarchical
    "CoordenadorGeral",
]
