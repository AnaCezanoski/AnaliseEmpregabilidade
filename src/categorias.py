"""Categorias do dataset do Kaggle e seus nomes em português."""

CATEGORIAS_PT = {
    "ACCOUNTANT": "Contabilidade",
    "ADVOCATE": "Direito",
    "AGRICULTURE": "Agricultura",
    "APPAREL": "Moda e Vestuário",
    "ARTS": "Artes",
    "AUTOMOBILE": "Automotivo",
    "AVIATION": "Aviação",
    "BANKING": "Bancos e Serviços Financeiros",
    "BPO": "Atendimento e Terceirização (BPO)",
    "BUSINESS-DEVELOPMENT": "Desenvolvimento de Negócios",
    "CHEF": "Gastronomia",
    "CONSTRUCTION": "Construção Civil",
    "CONSULTANT": "Consultoria",
    "DESIGNER": "Design",
    "DIGITAL-MEDIA": "Mídias Digitais",
    "ENGINEERING": "Engenharia",
    "FINANCE": "Finanças",
    "FITNESS": "Educação Física e Bem-estar",
    "HEALTHCARE": "Saúde",
    "HR": "Recursos Humanos",
    "INFORMATION-TECHNOLOGY": "Tecnologia da Informação",
    "PUBLIC-RELATIONS": "Relações Públicas e Comunicação",
    "SALES": "Vendas",
    "TEACHER": "Educação",
}


def nome_pt(categoria: str) -> str:
    """Devolve o nome em português de uma categoria do dataset."""
    return CATEGORIAS_PT.get(categoria, categoria.replace("-", " ").title())
