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

HABILIDADES = {
    "Atendimento ao público": "customer service client support",
    "Vendas": "sales selling retail customers",
    "Informática básica (Word, internet)": "computer skills microsoft office",
    "Planilhas (Excel)": "microsoft excel spreadsheets data entry",
    "Programação ou sites": "programming software development web",
    "Redes sociais e criação de conteúdo": "social media digital content marketing",
    "Design, edição de fotos ou vídeos": "graphic design photoshop video editing",
    "Cozinhar e preparar alimentos": "cooking food preparation kitchen",
    "Organização e rotinas de escritório": "administrative office organization scheduling",
    "Contas, caixa e controle de dinheiro": "cashier accounting bookkeeping finance",
    "Ensinar ou explicar coisas": "teaching tutoring education students",
    "Cuidar de pessoas (crianças, idosos)": "caregiving patient care healthcare",
    "Esportes e atividade física": "fitness sports training exercise",
    "Mecânica e consertos": "mechanic repair automotive maintenance",
    "Construção e reformas": "construction building renovation maintenance",
    "Arte, desenho ou música": "art drawing music creative",
    "Falar em público e comunicação": "communication public speaking presentation",
    "Agricultura ou jardinagem": "agriculture farming gardening",
    "Costura e moda": "sewing fashion apparel clothing",
    "Recrutar e organizar pessoas": "recruiting human resources coordination",
    "Trabalho em equipe": "teamwork collaboration",
    "Liderança": "leadership team leader",
}

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
        partes += [HABILIDADES[h] for h in self.habilidades if h in HABILIDADES]
        partes += [TIPOS_EXPERIENCIA.get(e.tipo, "") for e in self.experiencias]
        if any("Inglês" in i for i in self.idiomas):
            partes.append("english")
        # O texto livre em português também entra: nomes de ferramentas
        # (Excel, Python, Photoshop) costumam ser iguais nos dois idiomas.
        partes += [self.outras_habilidades, self.cursos, self.interesses]
        partes += [e.descricao for e in self.experiencias]
        return " ".join(p for p in partes if p)

    def vazio(self) -> bool:
        return not (self.habilidades or self.outras_habilidades or self.experiencias
                    or self.cursos or self.interesses)
