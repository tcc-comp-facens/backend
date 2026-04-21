"""
Utilitário de cruzamento de dados — despesas SIOPS × indicadores DataSUS.

Cruza despesas por subfunção com indicadores de saúde por tipo e ano,
produzindo pontos de dados cruzados para os agentes analíticos.

O mapeamento subfunção→indicador segue a tabela:
  301 → vacinacao
  302 → internacoes
  305 → dengue, covid
  mortalidade → transversal (cruza com todas as subfunções)

Requisitos: 9.4, 10.5
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Mapeamento subfunção → tipos de indicador
SUBFUNCAO_INDICADOR_MAP: dict[int, list[str]] = {
    301: ["vacinacao"],
    302: ["internacoes"],
    305: ["dengue", "covid"],
}

# Subfunções com as quais mortalidade cruza (transversal)
MORTALIDADE_SUBFUNCOES: list[int] = [301, 302, 303, 305]

SUBFUNCAO_NOMES: dict[int, str] = {
    301: "Atenção Básica",
    302: "Assistência Hospitalar",
    303: "Suporte Profilático",
    305: "Vigilância Epidemiológica",
}


def cross_domain_data(
    despesas: list[dict[str, Any]],
    indicadores: list[dict[str, Any]],
    date_from: int | None = None,
    date_to: int | None = None,
) -> list[dict[str, Any]]:
    """Cruza despesas com indicadores por subfunção e ano.

    Para cada subfunção no mapeamento, encontra indicadores do tipo
    correspondente no mesmo ano. Mortalidade é transversal — cruza
    com todas as subfunções (301, 302, 303, 305).

    Args:
        despesas: Lista de dicts com keys: subfuncao (int), subfuncaoNome (str),
                  ano (int), valor (float).
        indicadores: Lista de dicts com keys: tipo (str), ano (int), valor (float).
        date_from: Ano de início do período solicitado (para detecção de gaps).
        date_to: Ano de fim do período solicitado (para detecção de gaps).

    Returns:
        Lista de CrossedDataPoint dicts com keys: subfuncao, subfuncao_nome,
        tipo_indicador, ano, valor_despesa, valor_indicador.
    """
    if not despesas or not indicadores:
        return []

    crossed: list[dict[str, Any]] = []

    # Phase 1: Standard mapping (301→vacinacao, 302→internacoes, 305→dengue/covid)
    for subfuncao, tipos in SUBFUNCAO_INDICADOR_MAP.items():
        desp_by_year: dict[int, dict[str, Any]] = {}
        for d in despesas:
            if d.get("subfuncao") == subfuncao:
                desp_by_year[d["ano"]] = d

        for tipo in tipos:
            ind_by_year: dict[int, dict[str, Any]] = {}
            for i in indicadores:
                if i.get("tipo") == tipo:
                    ind_by_year[i["ano"]] = i

            common_years = sorted(set(desp_by_year) & set(ind_by_year))
            for year in common_years:
                d = desp_by_year[year]
                ind = ind_by_year[year]
                crossed.append({
                    "subfuncao": subfuncao,
                    "subfuncao_nome": d.get(
                        "subfuncaoNome",
                        SUBFUNCAO_NOMES.get(subfuncao, str(subfuncao)),
                    ),
                    "tipo_indicador": tipo,
                    "ano": year,
                    "valor_despesa": d["valor"],
                    "valor_indicador": ind["valor"],
                })

    # Phase 2: Mortalidade — transversal, crosses with ALL subfunções
    mort_by_year: dict[int, dict[str, Any]] = {}
    for i in indicadores:
        if i.get("tipo") == "mortalidade":
            mort_by_year[i["ano"]] = i

    if mort_by_year:
        for subfuncao in MORTALIDADE_SUBFUNCOES:
            desp_by_year = {}
            for d in despesas:
                if d.get("subfuncao") == subfuncao:
                    desp_by_year[d["ano"]] = d

            common_years = sorted(set(desp_by_year) & set(mort_by_year))
            for year in common_years:
                d = desp_by_year[year]
                ind = mort_by_year[year]
                crossed.append({
                    "subfuncao": subfuncao,
                    "subfuncao_nome": d.get(
                        "subfuncaoNome",
                        SUBFUNCAO_NOMES.get(subfuncao, str(subfuncao)),
                    ),
                    "tipo_indicador": "mortalidade",
                    "ano": year,
                    "valor_despesa": d["valor"],
                    "valor_indicador": ind["valor"],
                })

    logger.info("Crossed %d data points from %d despesas and %d indicadores",
                len(crossed), len(despesas), len(indicadores))

    return crossed


def detect_data_gaps(
    despesas: list[dict[str, Any]],
    indicadores: list[dict[str, Any]],
    date_from: int,
    date_to: int,
) -> dict[str, Any]:
    """Detecta lacunas nos dados disponíveis para o período solicitado.

    Verifica, para cada ano no intervalo [date_from, date_to], quais
    subfunções de despesa e quais tipos de indicador estão presentes
    ou ausentes. Retorna um relatório estruturado de cobertura.

    Args:
        despesas: Lista de despesas retornadas pelos agentes de domínio.
        indicadores: Lista de indicadores retornados pelos agentes de domínio.
        date_from: Ano de início do período solicitado.
        date_to: Ano de fim do período solicitado.

    Returns:
        Dict com:
        - expected_years: lista de anos no intervalo
        - despesas_coverage: cobertura por subfunção e ano
        - indicadores_coverage: cobertura por tipo e ano
        - gaps: lista de lacunas detectadas (descrição textual)
        - summary: resumo com contagens
    """
    expected_years = list(range(date_from, date_to + 1))
    all_subfuncoes = [301, 302, 303, 305]
    all_tipos = set()
    for tipos in SUBFUNCAO_INDICADOR_MAP.values():
        all_tipos.update(tipos)
    all_tipos.add("mortalidade")

    # Mapear dados disponíveis
    desp_available: dict[int, set[int]] = {}  # subfuncao → {anos}
    for d in despesas:
        sf = d.get("subfuncao", 0)
        ano = d.get("ano", 0)
        desp_available.setdefault(sf, set()).add(ano)

    ind_available: dict[str, set[int]] = {}  # tipo → {anos}
    for i in indicadores:
        tipo = i.get("tipo", "")
        ano = i.get("ano", 0)
        ind_available.setdefault(tipo, set()).add(ano)

    # Detectar gaps
    gaps: list[dict[str, Any]] = []

    # Despesas
    despesas_coverage: dict[int, dict[str, Any]] = {}
    for sf in all_subfuncoes:
        sf_nome = SUBFUNCAO_NOMES.get(sf, str(sf))
        available_years = desp_available.get(sf, set())
        missing_years = [y for y in expected_years if y not in available_years]
        present_years = [y for y in expected_years if y in available_years]

        despesas_coverage[sf] = {
            "subfuncao_nome": sf_nome,
            "present": present_years,
            "missing": missing_years,
            "coverage": len(present_years) / len(expected_years) if expected_years else 1.0,
        }

        if missing_years:
            gaps.append({
                "type": "despesa_missing",
                "subfuncao": sf,
                "subfuncao_nome": sf_nome,
                "missing_years": missing_years,
                "description": (
                    f"Despesa subfunção {sf} ({sf_nome}): "
                    f"sem dados para {', '.join(str(y) for y in missing_years)}"
                ),
            })

    # Indicadores
    indicadores_coverage: dict[str, dict[str, Any]] = {}
    for tipo in sorted(all_tipos):
        available_years = ind_available.get(tipo, set())
        missing_years = [y for y in expected_years if y not in available_years]
        present_years = [y for y in expected_years if y in available_years]

        indicadores_coverage[tipo] = {
            "present": present_years,
            "missing": missing_years,
            "coverage": len(present_years) / len(expected_years) if expected_years else 1.0,
        }

        if missing_years:
            gaps.append({
                "type": "indicador_missing",
                "tipo_indicador": tipo,
                "missing_years": missing_years,
                "description": (
                    f"Indicador {tipo}: "
                    f"sem dados para {', '.join(str(y) for y in missing_years)}"
                ),
            })

    # Cruzamentos impossíveis (subfunção tem dados mas indicador não, ou vice-versa)
    for sf, tipos in SUBFUNCAO_INDICADOR_MAP.items():
        sf_nome = SUBFUNCAO_NOMES.get(sf, str(sf))
        sf_years = desp_available.get(sf, set())
        for tipo in tipos:
            tipo_years = ind_available.get(tipo, set())
            # Anos onde só um lado tem dados
            only_despesa = sorted(sf_years - tipo_years)
            only_indicador = sorted(tipo_years - sf_years)
            if only_despesa:
                gaps.append({
                    "type": "cross_mismatch",
                    "subfuncao": sf,
                    "tipo_indicador": tipo,
                    "description": (
                        f"Cruzamento {sf_nome} × {tipo}: "
                        f"despesa sem indicador em {', '.join(str(y) for y in only_despesa)}"
                    ),
                })
            if only_indicador:
                gaps.append({
                    "type": "cross_mismatch",
                    "subfuncao": sf,
                    "tipo_indicador": tipo,
                    "description": (
                        f"Cruzamento {sf_nome} × {tipo}: "
                        f"indicador sem despesa em {', '.join(str(y) for y in only_indicador)}"
                    ),
                })

    # Resumo
    total_desp_cells = len(all_subfuncoes) * len(expected_years)
    total_ind_cells = len(all_tipos) * len(expected_years)
    present_desp = sum(
        len(desp_available.get(sf, set()) & set(expected_years))
        for sf in all_subfuncoes
    )
    present_ind = sum(
        len(ind_available.get(t, set()) & set(expected_years))
        for t in all_tipos
    )

    summary = {
        "period": f"{date_from}-{date_to}",
        "expected_years": expected_years,
        "despesas_completeness": round(present_desp / total_desp_cells, 4) if total_desp_cells else 1.0,
        "indicadores_completeness": round(present_ind / total_ind_cells, 4) if total_ind_cells else 1.0,
        "total_gaps": len(gaps),
        "has_gaps": len(gaps) > 0,
    }

    logger.info(
        "Data gap detection: %d gaps found, despesas %.0f%% complete, indicadores %.0f%% complete",
        len(gaps),
        summary["despesas_completeness"] * 100,
        summary["indicadores_completeness"] * 100,
    )

    return {
        "expected_years": expected_years,
        "despesas_coverage": despesas_coverage,
        "indicadores_coverage": indicadores_coverage,
        "gaps": gaps,
        "summary": summary,
    }
