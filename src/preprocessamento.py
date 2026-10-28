"""Limpeza de texto usada no treino e na previsão.

O mesmo pré-processamento precisa ser aplicado nos dois momentos; por isso
ele fica neste módulo e é chamado tanto pelo train.py quanto pelo app.py.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Palavras com marca de gênero (inglês). Elas são trocadas por um termo neutro
# antes do treino e da previsão, para que o modelo não aprenda associações
# entre gênero e área profissional (ver bias_test.py).
TERMOS_DE_GENERO = {
    "he", "she", "him", "her", "his", "hers", "himself", "herself",
    "mr", "mrs", "ms", "miss", "male", "female", "man", "woman", "men", "women",
    "boy", "girl", "boys", "girls", "husband", "wife", "father", "mother",
    "son", "daughter", "brother", "sister", "gentleman", "lady", "ladies",
    "boyfriend", "girlfriend", "maternity", "paternity", "fraternity", "sorority",
}

# A lista padrão de stopwords já remove alguns pronomes (he, she, his...).
# Tiramos os termos de gênero dela para que a neutralização seja uma escolha
# explícita, que o teste de viés consegue ligar e desligar.
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) - TERMOS_DE_GENERO

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+")
_NAO_LETRA = re.compile(r"[^a-z\s]")
_ESPACOS = re.compile(r"\s+")


def limpar_texto(texto: str, neutralizar_genero: bool = True) -> str:
    """Deixa o texto em minúsculas, remove ruído, stopwords e termos de gênero."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = _URL.sub(" ", texto)
    texto = _EMAIL.sub(" ", texto)
    texto = _NAO_LETRA.sub(" ", texto)
    tokens = []
    for tok in texto.split():
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        if neutralizar_genero and tok in TERMOS_DE_GENERO:
            continue
        tokens.append(tok)
    return _ESPACOS.sub(" ", " ".join(tokens)).strip()
