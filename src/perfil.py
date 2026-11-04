"""Opções do formulário guiado e montagem do perfil do jovem.

Cada habilidade e tipo de experiência tem um equivalente em inglês. Isso
permite montar um texto em inglês para o modelo (treinado com currículos em
inglês) mesmo quando a LLM não está disponível.
"""

from dataclasses import dataclass, field

from src.anonimizacao import anonimizar

ESCOLARIDADE = {
    "Ensino fundamental incompleto": "incomplete elementary school",
    "Ensino fundamental completo": "elementary school",
    "Ensino médio em andamento": "currently in high school",
    "Ensino médio completo": "high school diploma",
    "Curso técnico (em andamento ou completo)": "technical vocational course",
    "Ensino superior em andamento": "currently enrolled in college",
    "Ensino superior completo": "bachelor degree",
}

# Os termos em inglês usam o vocabulário que aparece nos currículos reais de cada
# área do dataset. Palavras ambíguas foram evitadas: "development", por exemplo,
# fazia "Programação" cair em Desenvolvimento de Negócios. Cada habilidade foi
# testada sozinha no modelo: a área esperada fica em 1º ou 2º lugar.
# Habilidades comportamentais (equipe, liderança) existem em qualquer área, então
# não mandam termos para o classificador; elas só vão para a LLM e o currículo.
HABILIDADES = {
    "Atendimento ao público": "bpo customer service process outsourcing voice process customer support representative",
    "Vendas": "sales retail sales associate selling customers store merchandise sales goals",
    "Informática básica (Word, internet)": "computer skills microsoft office word windows internet information technology computer support",
    "Planilhas (Excel)": "excel spreadsheets accounting reconciliation general ledger financial reports",
    "Programação ou sites": "information technology programming software programmer websites web applications coding",
    "Redes sociais e criação de conteúdo": "social media digital content marketing instagram posts online content",
    "Design, edição de fotos ou vídeos": "graphic design designer photoshop illustrator photo editing video editing visual",
    "Cozinhar e preparar alimentos": "cooking chef food preparation kitchen culinary meals",
    "Organização e rotinas de escritório": "administrative office clerical filing scheduling human resources records",
    "Contas, caixa e controle de dinheiro": "bank teller cash handling banking deposits transactions branch",
    "Ensinar ou explicar coisas": "teaching teacher tutoring students classroom lessons education",
    "Cuidar de pessoas (crianças, idosos)": "caregiving patient care healthcare elderly care childcare nursing",
    "Esportes e atividade física": "fitness sports personal training exercise coaching",
    "Mecânica e consertos": "automotive mechanic automobile vehicles car repair auto service technician dealership",
    "Construção e reformas": "construction building renovation carpentry masonry job site",
    "Arte, desenho ou música": "arts artist painting drawing music performance theater creative arts gallery",
    "Falar em público e comunicação": "public relations communications public speaking media relations press",
    "Agricultura ou jardinagem": "agriculture agricultural farming farm crops soil livestock harvest",
    "Costura e moda": "apparel clothing fashion design garments merchandising textiles sewing",
    "Recrutar e organizar pessoas": "human resources hr recruiting recruitment hiring onboarding employee relations",
    "Trabalho em equipe": "",
    "Liderança": "",
}

# Peso da habilidade favorita na média das previsões (as outras valem 1).
PESO_FAVORITA = 3

TIPOS_EXPERIENCIA = {
    "Emprego com carteira": "employment",
    "Trabalho informal ou bico": "freelance informal work",
    "Negócio da família": "family business",
    "Trabalho voluntário": "volunteer work",
    "Estágio ou jovem aprendiz": "internship apprentice",
    "Projeto da escola ou da comunidade": "school community project",
}

IDIOMAS = ["Inglês básico", "Inglês intermediário ou avançado", "Espanhol", "Libras"]


@dataclass
class Experiencia:
    tipo: str
    descricao: str
    duracao: str = ""


@dataclass
class Perfil:
    escolaridade: str
    habilidades: list[str] = field(default_factory=list)
    outras_habilidades: str = ""
    cursos: str = ""
    experiencias: list[Experiencia] = field(default_factory=list)
    interesses: str = ""
    idiomas: list[str] = field(default_factory=list)
    favorita: str = ""
    # Dados pessoais: nunca saem da sessão local.
    nome: str = ""
    email: str = ""
    telefone: str = ""
    cidade: str = ""

    def dados_pessoais(self) -> dict[str, str]:
        return {
            "[NOME]": self.nome, "[EMAIL]": self.email,
            "[TELEFONE]": self.telefone, "[CIDADE]": self.cidade,
        }

    def texto_anonimo(self) -> str:
        """Resumo do perfil em português, sem dados pessoais (vai para a LLM)."""
        nomes = [self.nome] if self.nome else []
        linhas = [f"Escolaridade: {self.escolaridade}"]
        if self.cursos:
            linhas.append(f"Cursos e formações: {anonimizar(self.cursos, nomes)}")
        if self.habilidades or self.outras_habilidades:
            hab = ", ".join(self.habilidades)
            if self.outras_habilidades:
                hab = f"{hab}; {anonimizar(self.outras_habilidades, nomes)}".strip("; ")
            linhas.append(f"Habilidades: {hab}")
        if self.favorita:
            linhas.append(f"Habilidade de que mais gosta: {self.favorita}")
        for i, exp in enumerate(self.experiencias, 1):
            dur = f" ({exp.duracao})" if exp.duracao else ""
            linhas.append(f"Experiência {i} – {exp.tipo}{dur}: {anonimizar(exp.descricao, nomes)}")
        if self.interesses:
            linhas.append(f"Interesses: {anonimizar(self.interesses, nomes)}")
        if self.idiomas:
            linhas.append(f"Idiomas: {', '.join(self.idiomas)}")
        return "\n".join(linhas)

    def texto_ingles_offline(self) -> str:
        """Texto em inglês montado só com o glossário, usado quando não há LLM."""
        partes = [ESCOLARIDADE.get(self.escolaridade, "")]
        partes.append(self.termos_glossario())
        partes += [TIPOS_EXPERIENCIA.get(e.tipo, "") for e in self.experiencias]
        if any("Inglês" in i for i in self.idiomas):
            partes.append("english")
        # O texto livre em português também entra: nomes de ferramentas
        # (Excel, Python, Photoshop) costumam ser iguais nos dois idiomas.
        partes += [self.outras_habilidades, self.cursos, self.interesses]
        partes += [e.descricao for e in self.experiencias]
        return " ".join(p for p in partes if p)

    def termos_glossario(self) -> str:
        """Termos fixos das habilidades marcadas.

        Também entram no modo com LLM, junto da tradução, para que o resultado
        não dependa das palavras que a LLM escolheu.
        """
        return " ".join(HABILIDADES[h] for h in self.habilidades if HABILIDADES.get(h))

    def partes_classificador(self) -> list[tuple[str, float]]:
        """Cada habilidade marcada como um trecho separado, com seu peso."""
        return [
            (HABILIDADES[h], PESO_FAVORITA if h == self.favorita else 1.0)
            for h in self.habilidades if HABILIDADES.get(h)
        ]

    def vazio(self) -> bool:
        return not (self.habilidades or self.outras_habilidades or self.experiencias
                    or self.cursos or self.interesses)
