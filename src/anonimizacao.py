"""Anonimização dos dados antes de qualquer envio para a LLM (LGPD).

Os dados pessoais (nome, e-mail, telefone, cidade) ficam só na sessão local.
A LLM recebe marcadores como [NOME] e [EMAIL], e o app troca os marcadores
pelos dados reais depois, já no computador do usuário.
"""

import re

PADROES = [
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("[CPF]", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("[CEP]", re.compile(r"\b\d{5}-?\d{3}\b")),
    ("[TELEFONE]", re.compile(r"(\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}[-\s]?\d{4}\b")),
    ("[LINK]", re.compile(r"https?://\S+|www\.\S+")),
]


def anonimizar(texto: str, nomes: list[str] | None = None) -> str:
    """Troca dados pessoais por marcadores no texto livre."""
    if not texto:
        return ""
    for marcador, padrao in PADROES:
        texto = padrao.sub(marcador, texto)
    for nome in nomes or []:
        for parte in {nome.strip(), *nome.split()}:
            if len(parte) >= 3:
                texto = re.sub(rf"\b{re.escape(parte)}\b", "[NOME]", texto, flags=re.IGNORECASE)
    return texto


def restaurar(texto: str, dados_pessoais: dict[str, str]) -> str:
    """Troca os marcadores pelos dados reais (só acontece localmente)."""
    for marcador, valor in dados_pessoais.items():
        if valor:
            texto = texto.replace(marcador, valor)
    return texto
