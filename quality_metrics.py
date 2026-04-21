"""
Métricas de qualidade e eficiência para comparação de topologias multiagente.

Módulo centralizado que calcula métricas complementares às já existentes
(tempo, CPU, memória, mensagens), cobrindo três eixos:

A. Eficiência dos Agentes:
   - A6: Overhead de coordenação (tempo supervisores / tempo total)
   - A7: Latency breakdown por fase (domínio / analítico / síntese)
   - A8: Communication efficiency (mensagens / agente)

B. Qualidade da Resposta:
   - B1: Deterministic consistency (outputs numéricos idênticos entre topologias)
   - B2: Faithfulness (texto reflete dados numéricos)
   - B3: Completeness (todos os achados relevantes mencionados no texto)
   - B4: Structural quality (texto contém seções esperadas)

C. Resiliência:
   - C1: Partial result coverage (agentes que completaram com sucesso)
   - C2: Graceful degradation score (qualidade sob falha simulada)

Requisitos: 11.1, 11.2, 11.3, 11.4
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Nomes das subfunções para verificação textual
SUBFUNCAO_NOMES: dict[int, str] = {
    301: "Atenção Básica",
    302: "Assistência Hospitalar",
    303: "Suporte Profilático",
    305: "Vigilância Epidemiológica",
}

# Tipos de agente por fase (para latency breakdown)
FASE_DOMINIO = {
    "vigilancia_epidemiologica",
    "saude_hospitalar",
    "atencao_primaria",
    "mortalidade",
}
FASE_ANALITICO = {"correlacao", "anomalias", "sintetizador"}
FASE_CONTEXTO = {"contexto_orcamentario"}
FASE_SUPERVISORES = {
    "supervisor_dominio",
    "supervisor_analitico",
    "supervisor_contexto",
}


# =========================================================================
# A. Eficiência dos Agentes
# =========================================================================


def compute_coordination_overhead(agent_metrics: list[dict]) -> dict[str, Any]:
    """A6 — Calcula o overhead de coordenação da topologia.

    Para a arquitetura hierárquica, mede quanto tempo é gasto em
    supervisores vs. agentes de trabalho. Para a estrela, o overhead
    é zero (orquestrador não aparece nas métricas de agentes).

    Args:
        agent_metrics: Lista de dicts com agentName e executionTimeMs.

    Returns:
        Dict com supervisor_time_ms, worker_time_ms, total_time_ms,
        overhead_ratio (0.0 a 1.0) e overhead_percent.
    """
    supervisor_time = 0.0
    worker_time = 0.0

    for m in agent_metrics:
        name = m.get("agentName", "")
        time_ms = m.get("executionTimeMs", 0)
        if name in FASE_SUPERVISORES:
            supervisor_time += time_ms
        else:
            worker_time += time_ms

    total = supervisor_time + worker_time
    ratio = supervisor_time / total if total > 0 else 0.0

    return {
        "supervisor_time_ms": round(supervisor_time, 2),
        "worker_time_ms": round(worker_time, 2),
        "total_time_ms": round(total, 2),
        "overhead_ratio": round(ratio, 4),
        "overhead_percent": round(ratio * 100, 2),
    }


def compute_latency_breakdown(agent_metrics: list[dict]) -> dict[str, Any]:
    """A7 — Calcula o breakdown de latência por fase do pipeline.

    Divide o tempo total em 4 fases: domínio, analítico, contexto
    e supervisores. Retorna tempo absoluto e percentual de cada fase.

    Args:
        agent_metrics: Lista de dicts com agentName e executionTimeMs.

    Returns:
        Dict com tempo e percentual por fase.
    """
    phases: dict[str, float] = {
        "dominio": 0.0,
        "analitico": 0.0,
        "contexto": 0.0,
        "supervisores": 0.0,
    }

    for m in agent_metrics:
        name = m.get("agentName", "")
        time_ms = m.get("executionTimeMs", 0)
        if name in FASE_DOMINIO:
            phases["dominio"] += time_ms
        elif name in FASE_ANALITICO:
            phases["analitico"] += time_ms
        elif name in FASE_CONTEXTO:
            phases["contexto"] += time_ms
        elif name in FASE_SUPERVISORES:
            phases["supervisores"] += time_ms

    total = sum(phases.values())

    breakdown: dict[str, Any] = {}
    for phase, time_ms in phases.items():
        pct = (time_ms / total * 100) if total > 0 else 0.0
        breakdown[phase] = {
            "time_ms": round(time_ms, 2),
            "percent": round(pct, 2),
        }
    breakdown["total_ms"] = round(total, 2)

    return breakdown


def compute_communication_efficiency(
    message_count: int,
    num_agents: int,
) -> dict[str, Any]:
    """A8 — Calcula a eficiência de comunicação.

    Mensagens por agente: quanto menor, mais eficiente a topologia
    em termos de overhead de comunicação.

    Args:
        message_count: Total de mensagens trocadas.
        num_agents: Número de agentes na topologia.

    Returns:
        Dict com messages_per_agent e total_messages.
    """
    ratio = message_count / num_agents if num_agents > 0 else 0.0
    return {
        "total_messages": message_count,
        "num_agents": num_agents,
        "messages_per_agent": round(ratio, 2),
    }


# =========================================================================
# B. Qualidade da Resposta
# =========================================================================


def compute_deterministic_consistency(
    star_result: dict[str, Any],
    hier_result: dict[str, Any],
) -> dict[str, Any]:
    """B1 — Verifica se ambas as topologias produzem resultados numéricos idênticos.

    Como ambas usam os mesmos agentes analíticos com os mesmos dados,
    correlações e anomalias devem ser idênticas. Divergências indicam
    bugs ou não-determinismo.

    Args:
        star_result: Resultado completo da topologia estrela.
        hier_result: Resultado completo da topologia hierárquica.

    Returns:
        Dict com flags de consistência e detalhes de divergências.
    """
    star_corr = star_result.get("correlacoes", [])
    hier_corr = hier_result.get("correlacoes", [])
    star_anom = star_result.get("anomalias", [])
    hier_anom = hier_result.get("anomalias", [])

    # Normalizar para comparação (ordenar por chave natural)
    def _sort_corr(corrs: list[dict]) -> list[tuple]:
        return sorted(
            (c.get("subfuncao", 0), c.get("tipo_indicador", ""),
             c.get("pearson", 0), c.get("spearman", 0), c.get("kendall", 0))
            for c in corrs
        )

    def _sort_anom(anoms: list[dict]) -> list[tuple]:
        return sorted(
            (a.get("subfuncao", 0), a.get("tipo_indicador", ""),
             a.get("ano", 0), a.get("tipo_anomalia", ""))
            for a in anoms
        )

    corr_identical = _sort_corr(star_corr) == _sort_corr(hier_corr)
    anom_identical = _sort_anom(star_anom) == _sort_anom(hier_anom)

    # Detalhar divergências se houver
    divergences: list[str] = []
    if not corr_identical:
        divergences.append(
            f"Correlações divergem: star={len(star_corr)}, hier={len(hier_corr)}"
        )
    if not anom_identical:
        divergences.append(
            f"Anomalias divergem: star={len(star_anom)}, hier={len(hier_anom)}"
        )

    all_identical = corr_identical and anom_identical

    return {
        "all_identical": all_identical,
        "correlacoes_identical": corr_identical,
        "anomalias_identical": anom_identical,
        "star_correlacoes_count": len(star_corr),
        "hier_correlacoes_count": len(hier_corr),
        "star_anomalias_count": len(star_anom),
        "hier_anomalias_count": len(hier_anom),
        "divergences": divergences,
    }


def compute_faithfulness(
    correlacoes: list[dict],
    anomalias: list[dict],
    texto: str,
) -> dict[str, Any]:
    """B2 — Verifica se o texto do sintetizador reflete os dados numéricos.

    Checklist automático: para cada correlação forte e cada anomalia,
    verifica se o texto menciona a subfunção e/ou indicador correspondente.

    Args:
        correlacoes: Lista de correlações calculadas.
        anomalias: Lista de anomalias detectadas.
        texto: Texto gerado pelo sintetizador.

    Returns:
        Dict com score (0.0 a 1.0), detalhes de hits/misses.
    """
    if not texto:
        return {
            "score": 0.0,
            "total_checkpoints": 0,
            "hits": 0,
            "misses": 0,
            "details": [],
        }

    texto_lower = texto.lower()
    hits = 0
    total = 0
    details: list[dict[str, Any]] = []

    # Verificar correlações fortes (classificação "alta")
    for c in correlacoes:
        if c.get("classificacao") != "alta":
            continue
        total += 1
        subfuncao = c.get("subfuncao", 0)
        tipo = c.get("tipo_indicador", "")
        subfuncao_nome = SUBFUNCAO_NOMES.get(subfuncao, str(subfuncao))

        # Verificar se subfunção OU nome OU tipo indicador aparece no texto
        found = (
            str(subfuncao) in texto
            or subfuncao_nome.lower() in texto_lower
            or tipo.lower() in texto_lower
        )
        if found:
            hits += 1
        details.append({
            "type": "correlacao_alta",
            "subfuncao": subfuncao,
            "tipo_indicador": tipo,
            "found": found,
        })

    # Verificar todas as anomalias
    for a in anomalias:
        total += 1
        subfuncao = a.get("subfuncao", 0)
        tipo = a.get("tipo_indicador", "")
        ano = a.get("ano", 0)
        tipo_anomalia = a.get("tipo_anomalia", "")
        subfuncao_nome = SUBFUNCAO_NOMES.get(subfuncao, str(subfuncao))

        # Verificar se ano E (subfunção OU nome OU tipo) aparecem no texto
        ano_found = str(ano) in texto
        entity_found = (
            str(subfuncao) in texto
            or subfuncao_nome.lower() in texto_lower
            or tipo.lower() in texto_lower
        )
        found = ano_found and entity_found
        if found:
            hits += 1
        details.append({
            "type": "anomalia",
            "subfuncao": subfuncao,
            "tipo_indicador": tipo,
            "ano": ano,
            "tipo_anomalia": tipo_anomalia,
            "found": found,
        })

    score = hits / total if total > 0 else 1.0

    return {
        "score": round(score, 4),
        "total_checkpoints": total,
        "hits": hits,
        "misses": total - hits,
        "details": details,
    }


def compute_faithfulness_llm(
    correlacoes: list[dict],
    anomalias: list[dict],
    contexto_orcamentario: dict,
    texto: str,
) -> dict[str, Any]:
    """B2 (LLM-as-judge) — Avalia faithfulness via chamada ao LLM.

    Envia os dados numéricos e o texto gerado para o LLM e pede
    uma avaliação de fidelidade numa escala de 1 a 5.

    Args:
        correlacoes: Lista de correlações calculadas.
        anomalias: Lista de anomalias detectadas.
        contexto_orcamentario: Dict com tendências orçamentárias.
        texto: Texto gerado pelo sintetizador.

    Returns:
        Dict com score (1-5), justificativa e método usado.
    """
    import llm_client

    prompt = (
        "Você é um avaliador de qualidade de textos analíticos. "
        "Avalie se o texto abaixo é FIEL aos dados numéricos fornecidos. "
        "O texto deve refletir corretamente as correlações, anomalias e "
        "tendências orçamentárias sem inventar informações.\n\n"
        f"DADOS DE CORRELAÇÃO:\n{correlacoes}\n\n"
        f"DADOS DE ANOMALIAS:\n{anomalias}\n\n"
        f"CONTEXTO ORÇAMENTÁRIO:\n{contexto_orcamentario}\n\n"
        f"TEXTO GERADO:\n{texto}\n\n"
        "Responda APENAS com um JSON no formato:\n"
        '{"score": <1-5>, "justificativa": "<explicação breve>"}\n\n'
        "Escala:\n"
        "1 = Texto contradiz os dados\n"
        "2 = Texto omite a maioria dos achados\n"
        "3 = Texto parcialmente fiel, com omissões significativas\n"
        "4 = Texto majoritariamente fiel, com omissões menores\n"
        "5 = Texto completamente fiel aos dados"
    )

    try:
        response = llm_client.generate(prompt)
        if response:
            # Tentar extrair JSON da resposta
            import json
            # Buscar JSON na resposta (pode vir com texto extra)
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "method": "llm_as_judge",
                    "score": parsed.get("score", 0),
                    "justificativa": parsed.get("justificativa", ""),
                    "raw_response": response,
                }
    except Exception as exc:
        logger.warning("Faithfulness LLM evaluation failed: %s", exc)

    return {
        "method": "llm_as_judge",
        "score": 0,
        "justificativa": "LLM indisponível para avaliação",
        "raw_response": None,
    }


def compute_completeness(
    correlacoes: list[dict],
    anomalias: list[dict],
    contexto_orcamentario: dict,
    texto: str,
) -> dict[str, Any]:
    """B3 — Verifica se todos os achados relevantes aparecem no texto.

    Diferente de faithfulness (que verifica se o que está no texto é
    verdadeiro), completeness verifica se TUDO que deveria estar no
    texto está lá.

    Args:
        correlacoes: Lista de correlações calculadas.
        anomalias: Lista de anomalias detectadas.
        contexto_orcamentario: Dict com tendências orçamentárias.
        texto: Texto gerado pelo sintetizador.

    Returns:
        Dict com score (0.0 a 1.0) e detalhes por categoria.
    """
    if not texto:
        return {
            "score": 0.0,
            "correlacoes_coverage": 0.0,
            "anomalias_coverage": 0.0,
            "contexto_coverage": 0.0,
            "details": {},
        }

    texto_lower = texto.lower()

    # 1. Cobertura de correlações (todas, não só as fortes)
    corr_total = len(correlacoes)
    corr_found = 0
    for c in correlacoes:
        subfuncao = c.get("subfuncao", 0)
        tipo = c.get("tipo_indicador", "")
        subfuncao_nome = SUBFUNCAO_NOMES.get(subfuncao, str(subfuncao))
        if (
            str(subfuncao) in texto
            or subfuncao_nome.lower() in texto_lower
            or tipo.lower() in texto_lower
        ):
            corr_found += 1

    # 2. Cobertura de anomalias
    anom_total = len(anomalias)
    anom_found = 0
    for a in anomalias:
        tipo_anomalia = a.get("tipo_anomalia", "")
        # Verificar menção do tipo de anomalia no texto
        if tipo_anomalia == "alto_gasto_baixo_resultado":
            keywords = ["alto gasto", "gasto acima", "ineficiência", "ineficiente"]
        else:
            keywords = ["baixo gasto", "gasto abaixo", "eficiência", "eficiente"]
        if any(kw in texto_lower for kw in keywords):
            anom_found += 1

    # 3. Cobertura de contexto orçamentário
    ctx_total = len(contexto_orcamentario)
    ctx_found = 0
    for subfuncao_key in contexto_orcamentario:
        subfuncao_nome = SUBFUNCAO_NOMES.get(
            int(subfuncao_key) if str(subfuncao_key).isdigit() else 0,
            str(subfuncao_key),
        )
        if (
            str(subfuncao_key) in texto
            or subfuncao_nome.lower() in texto_lower
        ):
            ctx_found += 1

    corr_cov = corr_found / corr_total if corr_total > 0 else 1.0
    anom_cov = anom_found / anom_total if anom_total > 0 else 1.0
    ctx_cov = ctx_found / ctx_total if ctx_total > 0 else 1.0

    # Score ponderado: correlações (40%), anomalias (40%), contexto (20%)
    score = corr_cov * 0.4 + anom_cov * 0.4 + ctx_cov * 0.2

    return {
        "score": round(score, 4),
        "correlacoes_coverage": round(corr_cov, 4),
        "anomalias_coverage": round(anom_cov, 4),
        "contexto_coverage": round(ctx_cov, 4),
        "details": {
            "correlacoes": {"found": corr_found, "total": corr_total},
            "anomalias": {"found": anom_found, "total": anom_total},
            "contexto": {"found": ctx_found, "total": ctx_total},
        },
    }


def compute_structural_quality(texto: str) -> dict[str, Any]:
    """B4 — Verifica se o texto contém as seções estruturais esperadas.

    O sintetizador deve produzir 4 seções:
    1. Resumo Executivo
    2. Análise das Correlações
    3. Discussão das Anomalias
    4. Contexto Orçamentário

    Args:
        texto: Texto gerado pelo sintetizador.

    Returns:
        Dict com score (0.0 a 1.0) e presença de cada seção.
    """
    if not texto:
        return {
            "score": 0.0,
            "sections_found": 0,
            "sections_expected": 4,
            "sections": {},
        }

    texto_lower = texto.lower()

    expected_sections = {
        "resumo_executivo": [
            "resumo executivo", "resumo", "executive summary",
        ],
        "correlacoes": [
            "correlações", "correlacoes", "análise das correlações",
            "analise das correlacoes", "correlação", "correlacao",
        ],
        "anomalias": [
            "anomalias", "discussão das anomalias", "anomalia",
            "ineficiências", "ineficiencias",
        ],
        "contexto_orcamentario": [
            "contexto orçamentário", "contexto orcamentario",
            "orçamentário", "orcamentario", "tendência", "tendencia",
        ],
    }

    sections: dict[str, bool] = {}
    found_count = 0

    for section_name, keywords in expected_sections.items():
        found = any(kw in texto_lower for kw in keywords)
        sections[section_name] = found
        if found:
            found_count += 1

    score = found_count / len(expected_sections)

    return {
        "score": round(score, 4),
        "sections_found": found_count,
        "sections_expected": len(expected_sections),
        "sections": sections,
    }


# =========================================================================
# C. Resiliência
# =========================================================================


def compute_partial_result_coverage(result: dict[str, Any]) -> dict[str, Any]:
    """C1 — Calcula a cobertura de resultados parciais.

    Verifica quantos componentes do resultado estão presentes e
    não-vazios, indicando quantos agentes completaram com sucesso.

    Args:
        result: Resultado completo de uma topologia.

    Returns:
        Dict com score (0.0 a 1.0) e status de cada componente.
    """
    components = {
        "despesas": bool(result.get("despesas")),
        "indicadores": bool(result.get("indicadores")),
        "dados_cruzados": bool(result.get("dados_cruzados")),
        "correlacoes": bool(result.get("correlacoes")),
        "anomalias": bool(result.get("anomalias")),
        "contexto_orcamentario": bool(result.get("contexto_orcamentario")),
        "texto_analise": bool(result.get("texto_analise")),
    }

    completed = sum(1 for v in components.values() if v)
    total = len(components)
    score = completed / total if total > 0 else 0.0

    return {
        "score": round(score, 4),
        "completed": completed,
        "total": total,
        "components": components,
    }


def compute_graceful_degradation(
    full_result: dict[str, Any],
    degraded_result: dict[str, Any],
) -> dict[str, Any]:
    """C2 — Calcula o score de degradação graciosa.

    Compara o resultado completo (sem falhas) com um resultado
    degradado (com falha simulada de agente) para medir quanto
    a qualidade cai.

    Args:
        full_result: Resultado sem falhas.
        degraded_result: Resultado com falha simulada.

    Returns:
        Dict com degradation_score (0.0 a 1.0, onde 1.0 = sem degradação).
    """
    full_coverage = compute_partial_result_coverage(full_result)
    degraded_coverage = compute_partial_result_coverage(degraded_result)

    full_score = full_coverage["score"]
    degraded_score = degraded_coverage["score"]

    # Degradation score: quanto do resultado original foi preservado
    preservation = degraded_score / full_score if full_score > 0 else 0.0

    # Comparar correlações e anomalias se ambos existirem
    corr_preserved = 1.0
    anom_preserved = 1.0

    full_corr = len(full_result.get("correlacoes", []))
    degraded_corr = len(degraded_result.get("correlacoes", []))
    if full_corr > 0:
        corr_preserved = degraded_corr / full_corr

    full_anom = len(full_result.get("anomalias", []))
    degraded_anom = len(degraded_result.get("anomalias", []))
    if full_anom > 0:
        anom_preserved = degraded_anom / full_anom

    return {
        "preservation_score": round(preservation, 4),
        "correlacoes_preserved": round(corr_preserved, 4),
        "anomalias_preserved": round(anom_preserved, 4),
        "full_coverage": full_coverage,
        "degraded_coverage": degraded_coverage,
    }


# =========================================================================
# Função agregadora — calcula todas as métricas de uma vez
# =========================================================================


def compute_all_quality_metrics(
    star_result: dict[str, Any],
    hier_result: dict[str, Any],
    star_agent_metrics: list[dict],
    hier_agent_metrics: list[dict],
    star_message_count: int,
    hier_message_count: int,
    use_llm_judge: bool = False,
) -> dict[str, Any]:
    """Calcula todas as métricas de qualidade e eficiência.

    Função de conveniência que agrega todas as métricas em um único
    dicionário, pronto para ser enviado via WebSocket ou persistido.

    Args:
        star_result: Resultado completo da topologia estrela.
        hier_result: Resultado completo da topologia hierárquica.
        star_agent_metrics: Métricas por agente da estrela.
        hier_agent_metrics: Métricas por agente da hierárquica.
        star_message_count: Total de mensagens da estrela.
        hier_message_count: Total de mensagens da hierárquica.
        use_llm_judge: Se True, também executa avaliação via LLM.

    Returns:
        Dict com todas as métricas organizadas por eixo.
    """
    # Número de agentes por topologia
    STAR_AGENTS = 8
    HIER_AGENTS = 11  # 8 agentes + 3 supervisores

    metrics: dict[str, Any] = {}

    # --- A. Eficiência ---
    metrics["efficiency"] = {
        "star": {
            "coordination_overhead": compute_coordination_overhead(
                star_agent_metrics
            ),
            "latency_breakdown": compute_latency_breakdown(star_agent_metrics),
            "communication_efficiency": compute_communication_efficiency(
                star_message_count, STAR_AGENTS
            ),
        },
        "hierarchical": {
            "coordination_overhead": compute_coordination_overhead(
                hier_agent_metrics
            ),
            "latency_breakdown": compute_latency_breakdown(hier_agent_metrics),
            "communication_efficiency": compute_communication_efficiency(
                hier_message_count, HIER_AGENTS
            ),
        },
    }

    # --- B. Qualidade da Resposta ---
    metrics["quality"] = {
        "deterministic_consistency": compute_deterministic_consistency(
            star_result, hier_result
        ),
        "star": {
            "faithfulness": compute_faithfulness(
                star_result.get("correlacoes", []),
                star_result.get("anomalias", []),
                star_result.get("texto_analise", ""),
            ),
            "completeness": compute_completeness(
                star_result.get("correlacoes", []),
                star_result.get("anomalias", []),
                star_result.get("contexto_orcamentario", {}),
                star_result.get("texto_analise", ""),
            ),
            "structural_quality": compute_structural_quality(
                star_result.get("texto_analise", ""),
            ),
        },
        "hierarchical": {
            "faithfulness": compute_faithfulness(
                hier_result.get("correlacoes", []),
                hier_result.get("anomalias", []),
                hier_result.get("texto_analise", ""),
            ),
            "completeness": compute_completeness(
                hier_result.get("correlacoes", []),
                hier_result.get("anomalias", []),
                hier_result.get("contexto_orcamentario", {}),
                hier_result.get("texto_analise", ""),
            ),
            "structural_quality": compute_structural_quality(
                hier_result.get("texto_analise", ""),
            ),
        },
    }

    # LLM-as-judge (opcional, consome chamada ao LLM)
    if use_llm_judge:
        metrics["quality"]["star"]["faithfulness_llm"] = (
            compute_faithfulness_llm(
                star_result.get("correlacoes", []),
                star_result.get("anomalias", []),
                star_result.get("contexto_orcamentario", {}),
                star_result.get("texto_analise", ""),
            )
        )
        metrics["quality"]["hierarchical"]["faithfulness_llm"] = (
            compute_faithfulness_llm(
                hier_result.get("correlacoes", []),
                hier_result.get("anomalias", []),
                hier_result.get("contexto_orcamentario", {}),
                hier_result.get("texto_analise", ""),
            )
        )

    # --- C. Resiliência ---
    metrics["resilience"] = {
        "star": compute_partial_result_coverage(star_result),
        "hierarchical": compute_partial_result_coverage(hier_result),
    }

    logger.info(
        "Quality metrics computed: consistency=%s, star_faithfulness=%.2f, "
        "hier_faithfulness=%.2f, star_completeness=%.2f, hier_completeness=%.2f",
        metrics["quality"]["deterministic_consistency"]["all_identical"],
        metrics["quality"]["star"]["faithfulness"]["score"],
        metrics["quality"]["hierarchical"]["faithfulness"]["score"],
        metrics["quality"]["star"]["completeness"]["score"],
        metrics["quality"]["hierarchical"]["completeness"]["score"],
    )

    return metrics


# =========================================================================
# Relatório comparativo textual — gerado após ambas as topologias
# =========================================================================


def generate_comparative_report(
    quality: dict[str, Any],
    star_agent_metrics: list[dict],
    hier_agent_metrics: list[dict],
    star_message_count: int,
    hier_message_count: int,
    data_coverage: dict[str, Any] | None = None,
) -> str:
    """Gera relatório textual comparativo entre as duas topologias.

    Produzido após ambas as topologias completarem, consolida todas
    as métricas de eficiência, qualidade e resiliência em texto
    legível para o usuário.

    Args:
        quality: Dict retornado por compute_all_quality_metrics().
        star_agent_metrics: Métricas por agente da estrela.
        hier_agent_metrics: Métricas por agente da hierárquica.
        star_message_count: Total de mensagens da estrela.
        hier_message_count: Total de mensagens da hierárquica.
        data_coverage: Dict com cobertura de dados e gaps detectados.

    Returns:
        Texto formatado do relatório comparativo.
    """
    sections: list[str] = []

    # ── Cabeçalho ──
    sections.append("=" * 60)
    sections.append("  RELATÓRIO COMPARATIVO — ESTRELA vs. HIERÁRQUICA")
    sections.append("=" * 60)
    sections.append("")

    # ── 1. Eficiência Operacional ──
    sections.append("━━━ 1. Cobertura de Dados ━━━")
    sections.append("")

    if data_coverage:
        summary = data_coverage.get("summary", {})
        gaps = data_coverage.get("gaps", [])
        desp_pct = summary.get("despesas_completeness", 1.0)
        ind_pct = summary.get("indicadores_completeness", 1.0)
        period = summary.get("period", "?")

        sections.append(f"  Período analisado: {period}")
        sections.append(f"  Completude das despesas SIOPS:     {desp_pct:.0%}")
        sections.append(f"  Completude dos indicadores DataSUS: {ind_pct:.0%}")
        sections.append("")

        if gaps:
            sections.append(f"  Lacunas detectadas ({len(gaps)}):")
            for g in gaps:
                sections.append(f"    ⚠ {g['description']}")
            sections.append("")
            sections.append(
                "  Nota: Correlações e anomalias foram calculadas apenas com"
            )
            sections.append(
                "  dados disponíveis. Anos sem dados foram excluídos."
            )
        else:
            sections.append("  ✓ Todos os dados esperados estão disponíveis.")
    else:
        sections.append("  Informação de cobertura não disponível.")
    sections.append("")

    sections.append("━━━ 2. Eficiência Operacional ━━━")
    sections.append("")

    eff = quality.get("efficiency", {})

    # Tempo total
    star_eff = eff.get("star", {})
    hier_eff = eff.get("hierarchical", {})
    star_total = star_eff.get("latency_breakdown", {}).get("total_ms", 0)
    hier_total = hier_eff.get("latency_breakdown", {}).get("total_ms", 0)

    sections.append(f"  Tempo total de execução:")
    sections.append(f"    Estrela:      {star_total:,.0f} ms")
    sections.append(f"    Hierárquica:  {hier_total:,.0f} ms")
    if star_total > 0 and hier_total > 0:
        diff_pct = ((hier_total - star_total) / star_total) * 100
        faster = "Estrela" if star_total < hier_total else "Hierárquica"
        sections.append(
            f"    → {faster} foi {abs(diff_pct):.1f}% mais rápida"
        )
    sections.append("")

    # Overhead de coordenação
    star_overhead = star_eff.get("coordination_overhead", {})
    hier_overhead = hier_eff.get("coordination_overhead", {})
    sections.append(f"  Overhead de coordenação:")
    sections.append(
        f"    Estrela:      {star_overhead.get('overhead_percent', 0):.1f}% "
        f"({star_overhead.get('supervisor_time_ms', 0):,.0f} ms em supervisores)"
    )
    sections.append(
        f"    Hierárquica:  {hier_overhead.get('overhead_percent', 0):.1f}% "
        f"({hier_overhead.get('supervisor_time_ms', 0):,.0f} ms em supervisores)"
    )
    sections.append("")

    # Latency breakdown
    sections.append(f"  Distribuição de tempo por fase:")
    for arch_name, arch_eff in [("Estrela", star_eff), ("Hierárquica", hier_eff)]:
        lb = arch_eff.get("latency_breakdown", {})
        sections.append(f"    {arch_name}:")
        for phase in ["dominio", "analitico", "contexto", "supervisores"]:
            phase_data = lb.get(phase, {})
            sections.append(
                f"      {phase.capitalize():15s} "
                f"{phase_data.get('time_ms', 0):>8,.0f} ms "
                f"({phase_data.get('percent', 0):>5.1f}%)"
            )
    sections.append("")

    # Comunicação
    star_comm = star_eff.get("communication_efficiency", {})
    hier_comm = hier_eff.get("communication_efficiency", {})
    sections.append(f"  Eficiência de comunicação:")
    sections.append(
        f"    Estrela:      {star_message_count} mensagens "
        f"({star_comm.get('messages_per_agent', 0):.1f} por agente)"
    )
    sections.append(
        f"    Hierárquica:  {hier_message_count} mensagens "
        f"({hier_comm.get('messages_per_agent', 0):.1f} por agente)"
    )
    sections.append("")

    # Métricas por agente
    sections.append(f"  Métricas por agente:")
    for arch_name, metrics_list in [
        ("Estrela", star_agent_metrics),
        ("Hierárquica", hier_agent_metrics),
    ]:
        sections.append(f"    {arch_name}:")
        for m in metrics_list:
            sections.append(
                f"      {m.get('agentName', '?'):30s} "
                f"{m.get('executionTimeMs', 0):>7,.0f} ms  "
                f"CPU {m.get('cpuPercent', 0):>5.1f}%  "
                f"Mem {m.get('memoryMb', 0):>6.1f} MB"
            )
    sections.append("")

    # ── 2. Qualidade da Resposta ──
    sections.append("━━━ 3. Qualidade da Resposta ━━━")
    sections.append("")

    qual = quality.get("quality", {})

    # Consistência determinística
    consistency = qual.get("deterministic_consistency", {})
    sections.append(f"  Consistência determinística:")
    if consistency.get("all_identical"):
        sections.append(
            "    ✓ Ambas as topologias produziram resultados numéricos idênticos"
        )
        sections.append(
            f"    ({consistency.get('star_correlacoes_count', 0)} correlações, "
            f"{consistency.get('star_anomalias_count', 0)} anomalias)"
        )
    else:
        sections.append(
            "    ✗ Divergências detectadas entre as topologias:"
        )
        for d in consistency.get("divergences", []):
            sections.append(f"      - {d}")
    sections.append("")

    # Faithfulness e Completeness
    for arch_name, arch_key in [("Estrela", "star"), ("Hierárquica", "hierarchical")]:
        arch_qual = qual.get(arch_key, {})
        faith = arch_qual.get("faithfulness", {})
        comp = arch_qual.get("completeness", {})
        struct = arch_qual.get("structural_quality", {})

        sections.append(f"  {arch_name}:")
        sections.append(
            f"    Fidelidade (faithfulness):    {faith.get('score', 0):.0%} "
            f"({faith.get('hits', 0)}/{faith.get('total_checkpoints', 0)} checkpoints)"
        )
        sections.append(
            f"    Completude (completeness):    {comp.get('score', 0):.0%} "
            f"(correlações {comp.get('correlacoes_coverage', 0):.0%}, "
            f"anomalias {comp.get('anomalias_coverage', 0):.0%}, "
            f"contexto {comp.get('contexto_coverage', 0):.0%})"
        )
        sections.append(
            f"    Qualidade estrutural:         {struct.get('score', 0):.0%} "
            f"({struct.get('sections_found', 0)}/{struct.get('sections_expected', 0)} seções)"
        )
    sections.append("")

    # LLM-as-judge (se disponível)
    for arch_name, arch_key in [("Estrela", "star"), ("Hierárquica", "hierarchical")]:
        llm_faith = qual.get(arch_key, {}).get("faithfulness_llm")
        if llm_faith and llm_faith.get("score", 0) > 0:
            sections.append(
                f"  {arch_name} — Avaliação LLM-as-judge: "
                f"{llm_faith['score']}/5 — {llm_faith.get('justificativa', '')}"
            )
    sections.append("")

    # ── 3. Resiliência ──
    sections.append("━━━ 4. Resiliência ━━━")
    sections.append("")

    resil = quality.get("resilience", {})
    for arch_name, arch_key in [("Estrela", "star"), ("Hierárquica", "hierarchical")]:
        arch_resil = resil.get(arch_key, {})
        sections.append(
            f"  {arch_name}: {arch_resil.get('score', 0):.0%} "
            f"({arch_resil.get('completed', 0)}/{arch_resil.get('total', 0)} componentes)"
        )
        components = arch_resil.get("components", {})
        missing = [k for k, v in components.items() if not v]
        if missing:
            sections.append(f"    Componentes ausentes: {', '.join(missing)}")
    sections.append("")

    # ── Conclusão ──
    sections.append("━━━ Conclusão ━━━")
    sections.append("")

    # Determinar vencedor por eixo
    star_wins = 0
    hier_wins = 0

    if star_total < hier_total:
        star_wins += 1
        sections.append("  • Eficiência: Estrela (menor tempo de execução)")
    elif hier_total < star_total:
        hier_wins += 1
        sections.append("  • Eficiência: Hierárquica (menor tempo de execução)")
    else:
        sections.append("  • Eficiência: Empate")

    star_faith = qual.get("star", {}).get("faithfulness", {}).get("score", 0)
    hier_faith = qual.get("hierarchical", {}).get("faithfulness", {}).get("score", 0)
    if star_faith > hier_faith:
        star_wins += 1
        sections.append("  • Qualidade: Estrela (maior fidelidade)")
    elif hier_faith > star_faith:
        hier_wins += 1
        sections.append("  • Qualidade: Hierárquica (maior fidelidade)")
    else:
        sections.append("  • Qualidade: Empate (mesma fidelidade)")

    if consistency.get("all_identical"):
        sections.append(
            "  • Consistência: Idêntica (ambas produzem os mesmos resultados numéricos)"
        )

    sections.append("")
    if star_wins > hier_wins:
        sections.append("  → Topologia Estrela apresentou melhor desempenho geral.")
    elif hier_wins > star_wins:
        sections.append("  → Topologia Hierárquica apresentou melhor desempenho geral.")
    else:
        sections.append("  → Ambas as topologias apresentaram desempenho equivalente.")

    sections.append("")
    sections.append("=" * 60)

    return "\n".join(sections)
