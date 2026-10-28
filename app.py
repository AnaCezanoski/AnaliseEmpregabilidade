"""Análise de Empregabilidade e Currículo – interface em Streamlit.

Rodar com:  streamlit run app.py
"""

import json
import os
import re
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src import llm, orientacao_offline
from src.anonimizacao import restaurar
from src.classificador import Classificador
from src.perfil import ESCOLARIDADE, HABILIDADES, IDIOMAS, TIPOS_EXPERIENCIA, Experiencia, Perfil

RAIZ = Path(__file__).parent

st.set_page_config(page_title="Análise de Empregabilidade e Currículo", layout="centered")

st.markdown(
    """
    <style>
    .aviso-ia {background:#eef2ff;border-left:4px solid #4c6ef5;padding:.8rem 1rem;
               border-radius:6px;margin-bottom:1rem;color:#1e2a5a;font-size:.95rem}
    .area-principal {font-size:1.6rem;font-weight:700;color:#4c6ef5;margin:.2rem 0 .6rem}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def carregar_classificador():
    return Classificador()


def carregar_metricas() -> dict | None:
    caminho = RAIZ / "reports" / "metricas.json"
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else None


def finalizar_curriculo(markdown: str, perfil: Perfil) -> str:
    """Coloca os dados pessoais de volta (localmente) e limpa marcadores vazios."""
    texto = restaurar(markdown, perfil.dados_pessoais())
    texto = re.sub(r"\[(NOME|EMAIL|TELEFONE|CIDADE|CPF|CEP|LINK)\]", "", texto)
    texto = re.sub(r"([ \t]*·[ \t]*){2,}", " · ", texto)
    texto = re.sub(r"^[ \t]*·[ \t]*|[ \t]*·[ \t]*$", "", texto, flags=re.MULTILINE)
    return texto.replace("# \n", "# Seu nome\n")


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Configuração")
    chave = st.text_input(
        f"Chave da API ({llm.provedor()})", type="password",
        help="Opcional. Sem chave, o app usa o modo offline, com feedback mais simples.",
    )
    if chave:
        os.environ[llm.CHAVES.get(llm.provedor(), "GEMINI_API_KEY")] = chave
    if llm.disponivel():
        st.success(f"LLM ativa: {llm.modelo()}")
    else:
        st.warning("Modo offline: sem LLM configurada.")

    metricas = carregar_metricas()
    if metricas:
        melhor = metricas["melhor_modelo"]
        m = metricas["modelos"][melhor]
        st.header("Sobre o modelo")
        st.caption(
            f"{melhor.replace('_', ' ').title()} treinado com {metricas['n_curriculos']} "
            f"currículos em {metricas['n_areas']} áreas."
        )
        c1, c2 = st.columns(2)
        c1.metric("Acurácia", f"{m['acuracia']:.0%}")
        c2.metric("F1 macro", f"{m['f1_macro']:.0%}")

    st.header("Privacidade")
    st.caption(
        "Seu nome, e-mail, telefone e cidade não são enviados para a IA nem salvos. "
        "Eles só entram no currículo no final, aqui no seu computador."
    )

# ---------------------------------------------------------------- cabeçalho
st.title("Análise de Empregabilidade e Currículo")
st.subheader("Descubra áreas que combinam com você e monte seu primeiro currículo")
st.markdown(
    '<div class="aviso-ia">Este assistente usa <b>inteligência artificial</b>. '
    "As sugestões são um ponto de partida para você pensar, não uma decisão. "
    "A IA pode errar: confira tudo antes de usar.</div>",
    unsafe_allow_html=True,
)

try:
    classificador = carregar_classificador()
except FileNotFoundError as erro:
    st.error(str(erro))
    st.stop()

# ---------------------------------------------------------------- formulário
st.markdown("### 1. Conte um pouco sobre você")
n_exp = st.number_input(
    "Quantas experiências você quer contar? Vale bico, ajuda em negócio da família, "
    "voluntariado ou projeto da escola.",
    min_value=0, max_value=4, value=1, step=1,
)

with st.form("formulario"):
    st.markdown("**Seus dados** (só usados no currículo final, não vão para a IA)")
    c1, c2 = st.columns(2)
    nome = c1.text_input("Nome", key="nome")
    cidade = c2.text_input("Cidade", key="cidade")
    email = c1.text_input("E-mail", key="email")
    telefone = c2.text_input("Telefone", key="telefone")

    st.markdown("**Formação**")
    escolaridade = st.selectbox("Escolaridade", list(ESCOLARIDADE))
    cursos = st.text_area("Cursos que você fez ou está fazendo (opcional)",
                          placeholder="Ex.: curso de informática no CRAS, curso de barista on-line")

    st.markdown("**Habilidades**")
    habilidades = st.multiselect("O que você sabe fazer bem?", list(HABILIDADES), key="habilidades")
    outras = st.text_input("Outras habilidades (opcional)",
                           placeholder="Ex.: Canva, conserto de celular, fazer unhas")

    experiencias = []
    if n_exp:
        st.markdown("**Experiências**")
    for i in range(int(n_exp)):
        c1, c2 = st.columns([2, 1])
        tipo = c1.selectbox(f"Tipo da experiência {i + 1}", list(TIPOS_EXPERIENCIA), key=f"t{i}")
        duracao = c2.text_input("Por quanto tempo?", key=f"d{i}", placeholder="Ex.: 6 meses")
        descricao = st.text_area(
            f"O que você fazia na experiência {i + 1}?", key=f"e{i}",
            placeholder="Ex.: ajudava minha tia na lanchonete, atendia clientes e cuidava do caixa",
        )
        if descricao.strip():
            experiencias.append(Experiencia(tipo, descricao.strip(), duracao.strip()))

    st.markdown("**Interesses**")
    interesses = st.text_area("Do que você gosta ou tem curiosidade? (opcional)",
                              placeholder="Ex.: gosto de mexer no computador e de desenhar")
    idiomas = st.multiselect("Idiomas (opcional)", IDIOMAS)

    enviar = st.form_submit_button("Ver minha orientação", type="primary", use_container_width=True)

# ---------------------------------------------------------------- processamento
if enviar:
    perfil = Perfil(
        escolaridade=escolaridade, habilidades=habilidades, outras_habilidades=outras.strip(),
        cursos=cursos.strip(), experiencias=experiencias, interesses=interesses.strip(),
        idiomas=idiomas, nome=nome.strip(), email=email.strip(),
        telefone=telefone.strip(), cidade=cidade.strip(),
    )
    if perfil.vazio():
        st.warning("Marque pelo menos uma habilidade ou conte uma experiência para continuar.")
        st.stop()

    perfil_anonimo = perfil.texto_anonimo()
    avisos = []
    with st.spinner("Analisando seu perfil..."):
        # 1. Texto em inglês para o classificador (o dataset é em inglês).
        texto_ingles, via_llm = perfil.texto_ingles_offline(), False
        if llm.disponivel():
            try:
                texto_ingles, via_llm = llm.traduzir_para_classificador(perfil_anonimo), True
            except llm.ErroLLM as erro:
                avisos.append(f"Não consegui usar a IA para preparar o texto ({erro}).")

        # 2. Classificação com o modelo de ML.
        areas = classificador.prever(texto_ingles, top_k=3)
        termos = classificador.termos_relevantes(texto_ingles, areas[0]["codigo"])

        # 3. Orientação gerada pela LLM (ou por regras, no modo offline).
        orientacao, modo = None, "llm"
        if llm.disponivel():
            try:
                orientacao = llm.gerar_orientacao(perfil_anonimo, areas, termos)
            except llm.ErroLLM as erro:
                avisos.append(f"A IA não respondeu ({erro}).")
        if orientacao is None:
            orientacao, modo = orientacao_offline.gerar(perfil, areas, termos), "offline"

    st.session_state["resultado"] = {
        "perfil": perfil, "areas": areas, "termos": termos, "orientacao": orientacao,
        "modo": modo, "avisos": avisos, "perfil_anonimo": perfil_anonimo,
        "texto_ingles": texto_ingles, "via_llm": via_llm,
    }

# ---------------------------------------------------------------- resultado
res = st.session_state.get("resultado")
if res:
    perfil, areas, orientacao = res["perfil"], res["areas"], res["orientacao"]
    st.divider()
    st.markdown("### 2. Sua orientação")
    if res["avisos"]:
        st.info(res["avisos"][-1] + " Usei o modo offline, com um feedback mais simples.")

    st.markdown("Área que mais combina com o seu perfil:")
    st.markdown(f'<div class="area-principal">{areas[0]["nome"]}</div>', unsafe_allow_html=True)
    for area in areas:
        st.progress(area["probabilidade"], text=f"{area['nome']} – {area['probabilidade']:.0%}")
    st.caption(
        "As porcentagens mostram o quanto o seu perfil se parece com currículos de cada área "
        "no conjunto de dados usado para treinar o modelo. Não medem a sua capacidade."
    )

    st.markdown("#### Por que essa sugestão?")
    st.write(orientacao["explicacao"])
    if res["termos"]:
        st.caption("Palavras que mais pesaram para o modelo: " + ", ".join(res["termos"]))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Seus pontos fortes")
        for item in orientacao["pontos_fortes"]:
            st.markdown(f"- {item}")
    with c2:
        st.markdown("#### O que melhorar")
        for item in orientacao["melhorias"]:
            st.markdown(f"- {item}")

    st.markdown("#### Próximos passos")
    for item in orientacao["proximos_passos"]:
        st.markdown(f"- {item}")

    st.markdown("### 3. Seu rascunho de currículo")
    curriculo = finalizar_curriculo(orientacao["curriculo_markdown"], perfil)
    with st.container(border=True):
        st.markdown(curriculo)
    st.download_button("Baixar currículo (.md)", curriculo, file_name="meu_curriculo.md",
                       mime="text/markdown", use_container_width=True)
    st.caption("Revise o rascunho com calma: confira datas, nomes e informações antes de enviar.")

    with st.expander("Transparência: o que foi enviado para a IA"):
        st.markdown("**Perfil anonimizado** (sem nome, e-mail, telefone e cidade):")
        st.code(res["perfil_anonimo"], language=None)
        origem = "gerado pela LLM" if res["via_llm"] else "montado pelo glossário (modo offline)"
        st.markdown(f"**Texto usado pelo modelo de classificação** ({origem}):")
        st.code(res["texto_ingles"], language=None)
        modelo_ml = classificador.nome.replace("_", " ").title()
        gerador = f"LLM ({llm.modelo()})" if res["modo"] == "llm" else "regras (modo offline)"
        st.caption(f"Classificação: {modelo_ml}. Orientação e currículo: {gerador}.")
