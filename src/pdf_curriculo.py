"""Gera o PDF do currículo a partir do Markdown produzido pela LLM (ou pelas regras).

Entende o Markdown simples que usamos no currículo: título (#), seções (##),
subtítulos (###), listas (- ou *) e negrito (**texto**).
"""

import re

from fpdf import FPDF

TINTA = (23, 26, 46)
SUAVE = (91, 96, 120)
VIOLETA = (109, 40, 217)
LINHA = (220, 216, 240)

# As fontes padrão do PDF só aceitam caracteres latin-1: trocamos os mais comuns
# que ficam fora (travessões, aspas curvas) e removemos o resto (ex.: emojis).
_TROCAS = {"–": "-", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'", "…": "...", "•": "-"}


def _limpar(texto: str) -> str:
    for de, para in _TROCAS.items():
        texto = texto.replace(de, para)
    texto = texto.replace("--", "-")  # "--" é tachado no markdown do fpdf
    texto = re.sub(r"(?<!\*)\*(?!\*)([^*]+)(?<!\*)\*(?!\*)", r"\1", texto)  # *itálico* -> texto
    texto = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", texto)  # [link](url) -> link
    return texto.encode("latin-1", "ignore").decode("latin-1").strip()


def gerar_pdf(markdown: str) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_page()
    largura = pdf.w - pdf.l_margin - pdf.r_margin

    # Faixa colorida no topo.
    pdf.set_fill_color(*VIOLETA)
    pdf.rect(0, 0, pdf.w, 4, style="F")

    linhas = markdown.splitlines()
    antes_das_secoes = True  # entre o nome e a primeira seção ficam os contatos
    for bruta in linhas:
        linha = bruta.rstrip()
        if not linha.strip():
            pdf.ln(1.5)
            continue

        if linha.startswith("# "):  # nome
            pdf.set_font("Helvetica", "B", 22)
            pdf.set_text_color(*TINTA)
            pdf.multi_cell(largura, 10, _limpar(linha[2:]), new_x="LMARGIN", new_y="NEXT")
            continue

        if linha.startswith("## "):
            antes_das_secoes = False
        elif antes_das_secoes:  # linha de contatos logo abaixo do nome
            pdf.set_font("Helvetica", size=10)
            pdf.set_text_color(*SUAVE)
            pdf.multi_cell(largura, 5.5, _limpar(linha.replace("**", "")), new_x="LMARGIN", new_y="NEXT")
            continue

        if linha.startswith("## "):  # seção
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*VIOLETA)
            pdf.set_char_spacing(1)
            pdf.cell(largura, 7, _limpar(linha[3:]).upper(), new_x="LMARGIN", new_y="NEXT")
            pdf.set_char_spacing(0)
            pdf.set_draw_color(*LINHA)
            pdf.set_line_width(0.4)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(2.5)
            continue

        if linha.startswith("### "):  # subtítulo (ex.: cargo)
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*TINTA)
            pdf.multi_cell(largura, 6, _limpar(linha[4:].replace("**", "")), new_x="LMARGIN", new_y="NEXT")
            continue

        pdf.set_font("Helvetica", size=10.5)
        pdf.set_text_color(*TINTA)
        item = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)", linha)
        if item:  # item de lista; subitens (com recuo no Markdown) ganham recuo e marcador menor
            nivel = min(len(item.group(1).expandtabs(4)) // 2, 2)
            x_marcador = pdf.l_margin + 1 + nivel * 5
            recuo = 5 + nivel * 5
            y = pdf.get_y()
            pdf.set_fill_color(*(VIOLETA if nivel == 0 else SUAVE))
            tamanho = 1.6 if nivel == 0 else 1.2
            pdf.ellipse(x_marcador, y + 2.8 - tamanho / 2, tamanho, tamanho, style="F")
            pdf.set_x(pdf.l_margin + recuo)
            pdf.multi_cell(largura - recuo, 5.6, _limpar(item.group(3)), markdown=True,
                           new_x="LMARGIN", new_y="NEXT")
            pdf.ln(0.8)
        else:
            pdf.multi_cell(largura, 5.6, _limpar(linha), markdown=True,
                           new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())
