"""Análise de Empregabilidade e Currículo – interface em Streamlit.

Rodar com:  streamlit run app.py
"""

import json
import os
import re
from html import escape
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src import llm, orientacao_offline
from src.anonimizacao import restaurar
from src.categorias import CATEGORIAS_PT
from src.classificador import Classificador
from src.pdf_curriculo import gerar_pdf
from src.perfil import ESCOLARIDADE, HABILIDADES, IDIOMAS, TIPOS_EXPERIENCIA, Experiencia, Perfil

RAIZ = Path(__file__).parent
SEM_FAVORITA = "Ainda não sei / tanto faz"

# Marca fictícia do projeto: troque aqui para mudar em toda a interface.
EMPRESA = "Primeiro Passo"
SLOGAN = "Carreiras para jovens"

st.set_page_config(page_title=f"{EMPRESA} · {SLOGAN}", page_icon="🚀",
                   layout="wide", initial_sidebar_state="collapsed")
st.html(f"<style>{(RAIZ / 'assets' / 'estilo.css').read_text(encoding='utf-8')}</style>")


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


def barras(areas: list[tuple[str, float]]) -> str:
    return "".join(
        f'<div class="barra"><div class="barra-info"><span>{escape(nome)}</span>'
        f"<span>{p:.0%}</span></div><div class=\"barra-trilho\">"
        f'<div class="barra-cheia" style="width:{max(p, .03):.0%}"></div></div></div>'
        for nome, p in areas
    )


def bloco(titulo: str, itens: list[str]) -> str:
    lis = "".join(f"<li>{escape(str(i))}</li>" for i in itens)
    return f'<div class="bloco"><h4>{titulo}</h4><ul>{lis}</ul></div>'


def pergunta(num: int, texto: str, dica: str = "") -> None:
    dica_html = f"<small>{dica}</small>" if dica else ""
    st.html(f'<div class="pergunta"><div class="pergunta-num">{num}</div>'
            f'<div class="pergunta-txt">{texto}{dica_html}</div></div>')


LOGO = (f'<div class="logo"><div class="logo-icone">{EMPRESA[0]}</div>'
        f"<div>{EMPRESA}<small>{SLOGAN}</small></div></div>")

# ---------------------------------------------------------------- barra lateral (configuração técnica)
with st.sidebar:
    st.header("Configuração")
    chave = st.text_input(
        f"Chave da API ({llm.provedor()})", type="password",
        help="Opcional. Sem chave, o app usa o modo offline, com feedback mais simples.",
    )
    if chave:
        os.environ[llm.CHAVES.get(llm.provedor(), "GEMINI_API_KEY")] = chave
    if llm.disponivel():
        st.success(f"IA ativa: {llm.modelo()}")
    else:
        st.warning("Modo offline: sem IA configurada.")

metricas = carregar_metricas()
n_curriculos = metricas["n_curriculos"] if metricas else 2400
n_areas = metricas["n_areas"] if metricas else len(CATEGORIAS_PT)

# ---------------------------------------------------------------- hero
st.html(f"""
<div class="hero">
  <div class="nav">
    {LOGO}
    <div class="nav-links">
      <a href="#como-funciona">Como funciona</a>
      <a href="#duvidas">Dúvidas</a>
      <a class="nav-cta" href="#perfil">Começar</a>
    </div>
  </div>
  <div class="hero-grid">
    <div>
      <div class="selo">✦ 100% gratuito · feito para quem está começando</div>
      <h1>Descubra onde você <span class="destaque">brilha</span> no mercado de trabalho.</h1>
      <p class="hero-sub">Conte o que você já sabe fazer, mesmo que seja bico, ajuda em casa ou
      projeto da escola. Em 2 minutos você recebe as áreas que mais combinam com você,
      um plano de ação e um currículo pronto.</p>
      <div class="botoes">
        <a class="btn-principal" href="#perfil">Fazer minha análise grátis →</a>
        <a class="btn-secundario" href="#como-funciona">Como funciona</a>
      </div>
      <div class="confianca">
        <span>⏱ 2 minutos</span><span>🔒 Sem cadastro</span><span>📄 Currículo para baixar</span>
      </div>
    </div>
    <div>
      <div class="mock">
        <div class="mock-topo"><span class="mock-rotulo">Seu match</span><span class="mock-tag">Exemplo</span></div>
        <div class="mock-area">🎨 <span>Design</span></div>
        {barras([("Design", .78), ("Mídias Digitais", .64), ("Artes", .41)])}
        <div class="mock-cv">📄 Currículo <b>pronto ✓</b></div>
      </div>
    </div>
  </div>
</div>
""")

itens = "".join(f'<span class="faixa-item">{escape(a)}</span>' for a in CATEGORIAS_PT.values())
st.html(f'<div class="faixa"><div class="faixa-trilho">{itens}{itens}</div></div>')

st.html(f"""
<div class="numeros">
  <div class="numero"><b>{n_areas}</b><span>áreas profissionais analisadas</span></div>
  <div class="numero"><b>{f"{n_curriculos:,}".replace(",", ".")}</b><span>currículos reais usados no treino</span></div>
  <div class="numero"><b>R$ 0</b><span>para sempre, sem pegadinha</span></div>
  <div class="numero"><b>0</b><span>dados pessoais armazenados</span></div>
</div>
""")

# ---------------------------------------------------------------- como funciona
st.html("""
<div class="secao" id="como-funciona">
  <span class="secao-tag">Como funciona</span>
  <h2>Do zero ao currículo em 3 passos</h2>
  <p>Sem textão, sem cadastro. Você toca no que sabe fazer e a gente faz o resto.</p>
</div>
<div class="passos">
  <div class="passo"><div class="passo-icone">👆</div><small>PASSO 1</small>
    <h3>Conte sobre você</h3><p>Marque suas habilidades e, se quiser, conte uma experiência. Vale tudo: bico, voluntariado, negócio da família.</p></div>
  <div class="passo"><div class="passo-icone">🧠</div><small>PASSO 2</small>
    <h3>A IA analisa seu perfil</h3><p>Um modelo treinado com milhares de currículos encontra as áreas que mais se parecem com você.</p></div>
  <div class="passo"><div class="passo-icone">🚀</div><small>PASSO 3</small>
    <h3>Receba seu plano</h3><p>Pontos fortes, o que melhorar, próximos passos e um rascunho de currículo para baixar.</p></div>
</div>
""")

try:
    classificador = carregar_classificador()
except FileNotFoundError as erro:
    st.error(str(erro))
    st.stop()

# ---------------------------------------------------------------- formulário
st.html("""
<div class="secao" id="perfil">
  <span class="secao-tag">Sua análise</span>
  <h2>Bora montar seu perfil? ✨</h2>
  <p>Só as perguntas 1 e 2 são obrigatórias. O resto é opcional.</p>
</div>
""")

with st.container(key="sec_form"):
    with st.form("formulario", border=False):
        pergunta(1, "Qual é a sua escolaridade?")
        escolaridade = st.selectbox("Escolaridade", list(ESCOLARIDADE), label_visibility="collapsed")

        pergunta(2, "O que você sabe fazer bem?", "Toque em todas que tiverem a ver com você.")
        habilidades = st.pills("Habilidades", list(HABILIDADES), selection_mode="multi",
                               key="habilidades", label_visibility="collapsed") or []

        pergunta(3, "Qual delas você mais curte?",
                 "Opcional. A favorita ganha mais peso na análise.")
        favorita = st.selectbox("Favorita", [SEM_FAVORITA] + list(HABILIDADES),
                                key="favorita", label_visibility="collapsed")

        pergunta(4, "Sua principal experiência",
                 "Opcional. Vale bico, negócio da família, voluntariado ou projeto da escola.")
        c1, c2 = st.columns([2, 1])
        tipo1 = c1.selectbox("Tipo", list(TIPOS_EXPERIENCIA), key="t0")
        duracao1 = c2.text_input("Por quanto tempo?", key="d0", placeholder="Ex.: 6 meses")
        descricao1 = st.text_input("O que você fazia?", key="e0",
                                   placeholder="Ex.: atendia clientes e cuidava do caixa na lanchonete")

        with st.expander("➕  Quer uma análise mais precisa? Conte mais (opcional)"):
            outras = st.text_input("Outras habilidades", placeholder="Ex.: Canva, conserto de celular")
            cursos = st.text_input("Cursos", placeholder="Ex.: informática no CRAS, barista on-line")
            interesses = st.text_input("Do que você gosta?", placeholder="Ex.: computador e desenho")
            idiomas = st.pills("Idiomas", IDIOMAS, selection_mode="multi", key="idiomas") or []
            st.markdown("**Outra experiência**")
            c1, c2 = st.columns([2, 1])
            tipo2 = c1.selectbox("Tipo", list(TIPOS_EXPERIENCIA), key="t1")
            duracao2 = c2.text_input("Por quanto tempo?", key="d1", placeholder="Ex.: 1 ano")
            descricao2 = st.text_input("O que você fazia?", key="e1")

        with st.expander("👤  Seus dados para o currículo (opcional, não vão para a IA)"):
            c1, c2 = st.columns(2)
            nome = c1.text_input("Nome", key="nome")
            cidade = c2.text_input("Cidade", key="cidade")
            email = c1.text_input("E-mail", key="email")
            telefone = c2.text_input("Telefone", key="telefone")

        enviar = st.form_submit_button("Ver minha análise  →", type="primary", use_container_width=True)
        st.html('<p class="privado">🔒 Seu nome e contatos nunca são enviados para a IA nem salvos.</p>')

# ---------------------------------------------------------------- processamento
if enviar:
    experiencias = [
        Experiencia(tipo, desc.strip(), dur.strip())
        for tipo, desc, dur in [(tipo1, descricao1, duracao1), (tipo2, descricao2, duracao2)]
        if desc.strip()
    ]
    favorita = "" if favorita == SEM_FAVORITA else favorita
    if favorita and favorita not in habilidades:
        habilidades = [*habilidades, favorita]
    perfil = Perfil(
        escolaridade=escolaridade, habilidades=habilidades, favorita=favorita, outras_habilidades=outras.strip(),
        cursos=cursos.strip(), experiencias=experiencias, interesses=interesses.strip(),
        idiomas=idiomas, nome=nome.strip(), email=email.strip(),
        telefone=telefone.strip(), cidade=cidade.strip(),
    )
    if perfil.vazio():
        with st.container(key="sec_aviso"):
            st.warning("Marque pelo menos uma habilidade ou conte uma experiência para continuar.")
        st.stop()

    perfil_anonimo = perfil.texto_anonimo()
    avisos = []
    with st.spinner("Analisando seu perfil... ✨"):
        # 1. Texto em inglês para o classificador (o dataset é em inglês).
        texto_ingles, via_llm = perfil.texto_ingles_offline(), False
        if llm.disponivel():
            try:
                traducao = llm.traduzir_para_classificador(perfil_anonimo)
                texto_ingles = f"{traducao} {perfil.termos_glossario()}".strip()
                via_llm = True
            except llm.ErroLLM as erro:
                avisos.append(f"Não consegui usar a IA para preparar o texto ({erro}).")

        # 2. Classificação com o modelo de ML.
        areas = classificador.prever(texto_ingles, top_k=3, partes=perfil.partes_classificador())
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
    st.html("""
    <div class="secao" id="resultado">
      <span class="secao-tag">Resultado</span>
      <h2>Olha só o que encontramos 🎉</h2>
    </div>
    """)
    if enviar:
        # Rola a página até o resultado logo depois do envio.
        # O número muda a cada envio para o navegador rodar o script de novo; ele
        # tenta algumas vezes porque a página ainda se reorganiza após o envio.
        st.session_state["envios"] = st.session_state.get("envios", 0) + 1
        st.iframe(
            f"<script>/* envio {st.session_state['envios']} */ let n = 0;"
            "const t = setInterval(() => {"
            "  parent.document.getElementById('resultado')?.scrollIntoView({behavior: 'smooth'});"
            "  if (++n >= 4) clearInterval(t);"
            "}, 400);</script>",
            height=1,
        )
    with st.container(key="sec_resultado"):
        if res["avisos"]:
            st.info(res["avisos"][-1] + " Usei o modo offline, com um feedback mais simples.")

        st.html(f"""
        <div class="res-card">
          <small>A área que mais combina com você</small>
          <div class="res-area">{escape(areas[0]["nome"])}</div>
          {barras([(a["nome"], a["probabilidade"]) for a in areas])}
          <p class="res-nota">As porcentagens mostram o quanto seu perfil se parece com currículos
          de cada área. Elas não medem a sua capacidade.</p>
        </div>
        """)

        aba_motivo, aba_plano, aba_cv, aba_transp = st.tabs(
            ["💡 Por que essa área", "🚀 Plano de ação", "📄 Currículo", "🔍 Transparência"]
        )
        with aba_motivo:
            st.write(orientacao["explicacao"])
            if res["termos"]:
                chips = "".join(f'<span class="chip">{escape(t)}</span>' for t in res["termos"])
                st.html(f'<p style="margin:.4rem 0 0;font-weight:700">Palavras que mais pesaram '
                        f'para o modelo:</p><div class="chips">{chips}</div>')

        with aba_plano:
            c1, c2 = st.columns(2)
            c1.html(bloco("✅ Seus pontos fortes", orientacao["pontos_fortes"]))
            c2.html(bloco("📈 O que melhorar", orientacao["melhorias"]))
            st.html(bloco("🎯 Próximos passos", orientacao["proximos_passos"]))

        with aba_cv:
            curriculo = finalizar_curriculo(orientacao["curriculo_markdown"], perfil)
            with st.container(border=True):
                st.markdown(curriculo)
            c1, c2 = st.columns([2, 1])
            try:
                c1.download_button("📄 Baixar currículo em PDF", gerar_pdf(curriculo),
                                   file_name="meu_curriculo.pdf", mime="application/pdf",
                                   type="primary", use_container_width=True)
            except Exception as erro:  # o .md continua disponível se o PDF falhar
                c1.warning(f"Não consegui gerar o PDF ({erro}).")
            c2.download_button("Baixar em texto (.md)", curriculo, file_name="meu_curriculo.md",
                               mime="text/markdown", use_container_width=True)
            st.caption("Revise o rascunho com calma: confira datas, nomes e informações antes de enviar.")

        with aba_transp:
            st.markdown("**Perfil anonimizado** (sem nome, e-mail, telefone e cidade):")
            st.code(res["perfil_anonimo"], language=None)
            origem = "gerado pela IA" if res["via_llm"] else "montado pelo glossário (modo offline)"
            st.markdown(f"**Texto usado pelo modelo de classificação** ({origem}):")
            st.code(res["texto_ingles"], language=None)
            modelo_ml = classificador.nome.replace("_", " ").title()
            gerador = f"IA ({llm.modelo()})" if res["modo"] == "llm" else "regras (modo offline)"
            st.caption(f"Classificação: {modelo_ml}. Orientação e currículo: {gerador}.")

# ---------------------------------------------------------------- dúvidas
st.html("""
<div class="secao" id="duvidas">
  <span class="secao-tag">Dúvidas</span>
  <h2>Perguntas frequentes</h2>
</div>
""")
with st.container(key="sec_faq"):
    _, meio, _ = st.columns([1, 4, 1])
    with meio:
        with st.expander("É de graça mesmo?"):
            st.write("Sim. Não tem cadastro, assinatura nem cobrança.")
        with st.expander("Meus dados ficam salvos?"):
            st.write("Seu nome, e-mail, telefone e cidade não são enviados para a IA: eles só entram "
                     "no currículo no final. O restante do perfil vai para a IA do Google (Gemini) sem "
                     "esses dados, e o nosso app não guarda nada depois que você fecha a página. "
                     "Atenção: no plano gratuito, o Google pode usar o que recebe para melhorar os "
                     "produtos dele. Por isso, evite escrever nomes de outras pessoas ou empresas nos "
                     "campos de texto.")
        with st.expander("Nunca trabalhei. Serve para mim?"):
            st.write("Serve! Bico, ajuda em negócio da família, voluntariado, projeto da escola e cursos "
                     "contam. E se você ainda não tem nada disso, marque só as suas habilidades.")
        with st.expander("A IA decide qual profissão eu devo seguir?"):
            st.write("Não. A análise mostra com quais áreas o seu perfil se parece hoje. É um ponto de "
                     "partida para você pensar, e a IA pode errar. A decisão é sempre sua.")

# ---------------------------------------------------------------- rodapé
st.html(f"""
<div class="rodape">
  <div class="rodape-grid">
    <div>{LOGO}<p>Ajudamos jovens a descobrir suas áreas de interesse e a montar o primeiro
    currículo, de graça e sem guardar dados pessoais.</p></div>
    <div class="rodape-links"><b>Navegue</b>
      <a href="#como-funciona" style="color:#AEB2D4">Como funciona</a>
      <a href="#perfil" style="color:#AEB2D4">Fazer análise</a>
      <a href="#duvidas" style="color:#AEB2D4">Dúvidas</a>
    </div>
  </div>
  <div class="rodape-fim">© 2026 {EMPRESA} · Marca fictícia de um projeto acadêmico de Sistemas de
  Informação Inteligentes (PUCPR). Esta análise usa inteligência artificial e pode errar.</div>
</div>
""")
