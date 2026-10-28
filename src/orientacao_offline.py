"""Orientação por regras, usada quando a LLM não está disponível.

Garante que o sistema continue funcionando (por exemplo, se a internet ou a
cota da API falharem durante a demonstração). O resultado é mais simples que
o da LLM, e o app avisa o usuário disso.
"""

from src.perfil import Perfil


def gerar(perfil: Perfil, areas: list[dict], termos: list[str]) -> dict:
    principal = areas[0]["nome"]
    explicacao = (
        f"Pelas habilidades e experiências que você contou, o modelo encontrou mais "
        f"semelhança com currículos da área de {principal}."
    )
    if termos:
        explicacao += f" As palavras que mais pesaram foram: {', '.join(termos[:5])}."
    explicacao += " Isso é só um ponto de partida: você decide o caminho."

    fortes = []
    if perfil.experiencias:
        tipos = ", ".join(sorted({e.tipo.lower() for e in perfil.experiencias}))
        fortes.append(f"Você já tem experiência prática ({tipos}), e isso conta muito.")
    if len(perfil.habilidades) >= 3:
        fortes.append(f"Você marcou {len(perfil.habilidades)} habilidades, um perfil versátil.")
    if perfil.cursos:
        fortes.append("Você já buscou cursos e formação por conta própria.")
    if perfil.idiomas:
        fortes.append(f"Idiomas: {', '.join(perfil.idiomas)}.")
    if not fortes:
        fortes.append("Você deu o primeiro passo ao organizar o seu perfil.")

    melhorias = []
    if not perfil.experiencias:
        melhorias.append("Inclua qualquer experiência, mesmo informal, voluntária ou na escola.")
    if any(len(e.descricao.split()) < 6 for e in perfil.experiencias):
        melhorias.append("Descreva melhor o que você fazia em cada experiência, com exemplos.")
    if not perfil.cursos:
        melhorias.append("Um curso curto e gratuito na área ajuda a mostrar interesse.")
    if not perfil.email and not perfil.telefone:
        melhorias.append("Coloque pelo menos um contato (e-mail ou telefone) no currículo.")
    if not melhorias:
        melhorias.append("Revise o currículo e adapte o objetivo para cada vaga.")

    passos = [
        f"Procure vagas de entrada ou jovem aprendiz em {principal}.",
        "Busque cursos gratuitos em instituições como SENAI, SENAC, Sebrae ou a "
        "Escola Virtual da Fundação Bradesco.",
    ]
    if len(areas) > 1:
        passos.append(f"Vale também conhecer a área de {areas[1]['nome']}.")

    return {
        "explicacao": explicacao,
        "pontos_fortes": fortes,
        "melhorias": melhorias,
        "proximos_passos": passos,
        "curriculo_markdown": curriculo(perfil, principal),
    }


def curriculo(perfil: Perfil, area: str) -> str:
    linhas = ["# [NOME]", "[EMAIL] · [TELEFONE] · [CIDADE]", "",
              "## Objetivo", f"Oportunidade de entrada na área de {area}.", "",
              "## Formação", f"- {perfil.escolaridade}"]
    if perfil.cursos:
        linhas.append(f"- {perfil.cursos}")
    if perfil.experiencias:
        linhas += ["", "## Experiências"]
        for e in perfil.experiencias:
            dur = f" ({e.duracao})" if e.duracao else ""
            linhas.append(f"- **{e.tipo}**{dur}: {e.descricao}")
    habilidades = perfil.habilidades + ([perfil.outras_habilidades] if perfil.outras_habilidades else [])
    if habilidades:
        linhas += ["", "## Habilidades", *[f"- {h}" for h in habilidades]]
    if perfil.idiomas:
        linhas += ["", "## Idiomas", *[f"- {i}" for i in perfil.idiomas]]
    return "\n".join(linhas)
