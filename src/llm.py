"""Integração com a LLM.

O provedor é escolhido por variáveis de ambiente (ver .env.example):
    LLM_PROVIDER = gemini | openai | anthropic   (padrão: gemini)
    LLM_MODEL    = nome do modelo (opcional)
    GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY

Sem chave configurada, o app continua funcionando em modo offline, com
feedback gerado por regras (ver orientacao_offline no app).
"""

import json
import os
import re

from src import prompts

MODELOS_PADRAO = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
}
CHAVES = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class ErroLLM(RuntimeError):
    pass


def provedor() -> str:
    return os.getenv("LLM_PROVIDER", "gemini").strip().lower()


def modelo() -> str:
    return os.getenv("LLM_MODEL") or MODELOS_PADRAO.get(provedor(), "")


def disponivel() -> bool:
    return bool(os.getenv(CHAVES.get(provedor(), ""), "").strip())


def _chamar(sistema: str, mensagem: str, temperatura: float) -> str:
    nome = provedor()
    chave = os.getenv(CHAVES.get(nome, ""), "")
    if not chave:
        raise ErroLLM(f"Chave de API não configurada para o provedor '{nome}'.")
    try:
        if nome == "gemini":
            from google import genai
            from google.genai import types

            resposta = genai.Client(api_key=chave).models.generate_content(
                model=modelo(),
                contents=mensagem,
                config=types.GenerateContentConfig(
                    system_instruction=sistema,
                    temperature=temperatura,
                    response_mime_type="application/json",
                ),
            )
            return resposta.text
        if nome == "openai":
            from openai import OpenAI

            resposta = OpenAI(api_key=chave).chat.completions.create(
                model=modelo(),
                temperature=temperatura,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": sistema},
                          {"role": "user", "content": mensagem}],
            )
            return resposta.choices[0].message.content
        if nome == "anthropic":
            import anthropic

            resposta = anthropic.Anthropic(api_key=chave).messages.create(
                model=modelo(),
                max_tokens=3000,
                temperature=temperatura,
                system=sistema,
                messages=[{"role": "user", "content": mensagem}],
            )
            return "".join(b.text for b in resposta.content if b.type == "text")
    except ImportError as erro:
        raise ErroLLM(f"Biblioteca do provedor '{nome}' não instalada: {erro}") from erro
    except Exception as erro:  # erros de rede, cota, chave inválida...
        raise ErroLLM(f"Falha ao chamar a LLM ({nome}): {erro}") from erro
    raise ErroLLM(f"Provedor desconhecido: '{nome}'. Use gemini, openai ou anthropic.")


def _extrair_json(texto: str) -> dict:
    """Lê o JSON da resposta, mesmo que venha dentro de ```json ... ```."""
    texto = (texto or "").strip()
    bloco = re.search(r"\{.*\}", texto, re.DOTALL)
    if not bloco:
        raise ErroLLM("A LLM não devolveu JSON.")
    try:
        return json.loads(bloco.group(0))
    except json.JSONDecodeError as erro:
        raise ErroLLM(f"JSON inválido na resposta da LLM: {erro}") from erro


def traduzir_para_classificador(perfil_anonimo: str) -> str:
    """Converte o perfil (português) em texto de currículo em inglês."""
    dados = _extrair_json(_chamar(prompts.SISTEMA_TRADUCAO, perfil_anonimo, 0.0))
    texto = str(dados.get("english_resume", "")).strip()
    if not texto:
        raise ErroLLM("A LLM devolveu um texto vazio.")
    return texto


def gerar_orientacao(perfil_anonimo: str, areas: list[dict], termos: list[str]) -> dict:
    """Gera explicação, pontos fortes, melhorias, próximos passos e currículo."""
    mensagem = prompts.prompt_orientacao(perfil_anonimo, areas, termos)
    dados = _extrair_json(_chamar(prompts.SISTEMA_ORIENTACAO, mensagem, 0.4))
    for chave in ("pontos_fortes", "melhorias", "proximos_passos"):
        valor = dados.get(chave, [])
        dados[chave] = [str(v) for v in valor] if isinstance(valor, list) else [str(valor)]
    dados["explicacao"] = str(dados.get("explicacao", ""))
    dados["curriculo_markdown"] = str(dados.get("curriculo_markdown", ""))
    return dados
