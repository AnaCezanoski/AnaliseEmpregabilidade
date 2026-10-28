# Análise de Empregabilidade e Currículo – Assistente Inteligente para Jovens

Projeto do TDE2 de Sistemas de Informação Inteligentes (PUCPR).
Integrantes: Ana Julia Cezanoski, Eduardo Nicolosi e Francielly Pazello.

O jovem responde a um formulário guiado sobre formação, habilidades e experiências
e recebe: a área profissional mais compatível (com as 3 mais prováveis), a explicação
do resultado, pontos fortes, o que melhorar, próximos passos e um rascunho de currículo.

## Como rodar

```bash
pip install -r requirements.txt

# 1. Baixar o dataset (escolha uma opção)
#    a) automático:  python train.py --baixar
#    b) manual: baixe em https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
#       e coloque o arquivo Resume.csv em data/Resume.csv

# 2. Treinar e comparar os modelos
python train.py

# 3. (Opcional) Teste de viés
python bias_test.py

# 4. Configurar a LLM (opcional, mas recomendado)
cp .env.example .env        # e cole a chave do Gemini em GEMINI_API_KEY
#    chave gratuita em https://aistudio.google.com/apikey

# 5. Abrir o app
streamlit run app.py
```

Sem chave de API o app funciona em **modo offline**: a classificação continua igual,
mas o feedback e o currículo são gerados por regras simples. Isso evita que a demo
quebre se a internet ou a cota da API falharem. A chave também pode ser colada direto
na barra lateral do app.

## Arquitetura

| Etapa | Arquivo | O que faz |
| --- | --- | --- |
| Entrada | `app.py`, `src/perfil.py` | Formulário guiado (Streamlit) e montagem do perfil |
| Anonimização | `src/anonimizacao.py` | Troca nome, e-mail, telefone, CPF, CEP e links por marcadores |
| Preparação | `src/llm.py` | LLM converte o perfil em texto de currículo em inglês e neutro em gênero |
| Classificação | `src/classificador.py` | Modelo de ML prevê as 3 áreas mais prováveis e as palavras que mais pesaram |
| Orientação | `src/llm.py`, `src/prompts.py` | LLM gera explicação, feedback, próximos passos e currículo |
| Saída | `app.py` | Relatório, currículo com dados restaurados localmente e download |

Fluxo: formulário → anonimização → LLM (tradução) → modelo de ML → LLM (orientação) →
dados pessoais devolvidos localmente → relatório e currículo.

### Por que ML e LLM juntos?

- O **modelo de ML** dá um resultado mensurável e reproduzível: dá para calcular acurácia,
  F1 e matriz de confusão e comparar modelos.
- A **LLM** faz o que o ML não faz: traduz o perfil (o dataset é em inglês), transforma
  experiências informais em linguagem de currículo e explica o resultado em linguagem simples.

## Machine Learning (`train.py`)

- **Dados:** Resume Dataset (Kaggle), cerca de 2.400 currículos em 24 áreas.
- **Pré-processamento** (`src/preprocessamento.py`): minúsculas, remoção de links, e-mails,
  números e stopwords, e **neutralização de gênero** (remove he/she, his/her, man/woman...).
- **Vetorização:** TF-IDF com unigramas e bigramas (até 20 mil termos).
- **Modelos comparados:** Regressão Logística e Random Forest, ambos com
  `class_weight="balanced"` porque o dataset é desbalanceado.
- **Divisão:** 80% treino / 20% teste, estratificada.
- **Escolha:** o modelo com maior **F1 macro** é salvo em `models/modelo.joblib`.

Saídas em `reports/`: `metricas.json`, `relatorio_<modelo>.txt` (precisão, recall e F1
por área) e `matriz_confusao_<modelo>.png`.

## Teste de viés (`bias_test.py`)

Versões contrafactuais dos currículos de teste:

1. **Gênero:** troca he↔she, his↔her, man↔woman etc. e mede quantas previsões mudam,
   comparando um modelo sem neutralização com o modelo mitigado.
2. **Origem social:** acrescenta uma frase sobre escola pública e bairro de baixa renda
   ou escola particular e bairro nobre, e mede se a sugestão muda.

Resultado em `reports/vies.json`.

## Ética e privacidade (o que está implementado)

- **LGPD:** nada é salvo em disco. Nome, e-mail, telefone e cidade nunca vão para a LLM;
  o texto livre também é anonimizado por expressões regulares. Os dados só voltam ao
  currículo no final, localmente.
- **Viés:** neutralização de gênero no treino e na previsão, prompt que proíbe pronomes
  com gênero e teste contrafactual.
- **Transparência:** aviso de uso de IA, top 3 áreas com probabilidade, palavras que mais
  pesaram e uma seção mostrando exatamente o que foi enviado para a IA.
- **Impacto no usuário:** prompt com tom encorajador, que valoriza trabalho informal,
  voluntário e em negócio da família.
- **Limitações:** a LLM pode errar ou inventar; o dataset é americano e em inglês e não
  representa bem o mercado brasileiro nem áreas comuns de primeiro emprego; as
  porcentagens indicam semelhança com currículos do dataset, não capacidade da pessoa.

## Provedores de LLM

Configure em `.env` com `LLM_PROVIDER`:

| Provedor | Variável da chave | Modelo padrão |
| --- | --- | --- |
| `gemini` (padrão, tem camada gratuita) | `GEMINI_API_KEY` | `gemini-2.5-flash` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |

Para usar outro modelo, defina `LLM_MODEL`. Para OpenAI ou Anthropic, instale também
`openai` ou `anthropic` (estão comentados no `requirements.txt`).
