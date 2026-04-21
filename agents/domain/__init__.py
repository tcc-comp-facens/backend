"""Agentes de domínio especializados em fontes de dados de saúde."""

from agents.domain.vigilancia_epidemiologica import AgenteVigilanciaEpidemiologica
from agents.domain.saude_hospitalar import AgenteSaudeHospitalar
from agents.domain.atencao_primaria import AgenteAtencaoPrimaria
from agents.domain.mortalidade import AgenteMortalidade

__all__ = [
    "AgenteVigilanciaEpidemiologica",
    "AgenteSaudeHospitalar",
    "AgenteAtencaoPrimaria",
    "AgenteMortalidade",
]
