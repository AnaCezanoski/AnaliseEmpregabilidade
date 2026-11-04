# Análise de Empregabilidade e Currículo – Assistente Inteligente para Jovens

Projeto do TDE2 de Sistemas de Informação Inteligentes (PUCPR).
Integrantes: Ana Julia Cezanoski, Eduardo Nicolosi e Francielly Pazello.

O jovem responde a um formulário rápido (cerca de 2 minutos) sobre formação, habilidades
e experiências e recebe: as 3 áreas profissionais mais compatíveis, a explicação do
resultado, pontos fortes, o que melhorar, próximos passos e um currículo pronto para
baixar em **PDF**.

A interface usa a marca fictícia **"Primeiro Passo · Carreiras para jovens"**, criada para
a apresentação.

📄 **Justificativa ética completa, com os resultados dos testes de viés: [ETICA.md](ETICA.md).**

## Como rodar

```bash
pip install -r requirements.txt

# 1. Baixar o dataset (escolha uma opção)
#    a) automático:  python train.py --baixar
#    b) manual: baixe em https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
#       e coloque o arquivo Resume.csv em data/Resume.csv

# 2. (Opcional) Treinar e comparar os modelos
#    O repositório já traz o modelo treinado em models/modelo.joblib.
#    Rodar de novo sobrescreve o modelo e as métricas em reports/.
python train.py

# 3. (Opcional) Testes de viés: gera reports/vies.json (não altera o modelo)
python bias_test.py

# 4. Configurar a LLM (opcional, mas recomendado)
cp .env.example .env        # e cole a chave do Gemini em GEMINI_API_KEY
#    chave gratuita em https://aistudio.google.com/apikey

# 5. Abrir o app
streamlit run app.py
```

Sem chave de API o app funciona em **modo offline**: a classificação continua igual,
mas o feedback e o currículo são gerados por regras simples. Isso evita que a demo
quebre se a internet ou a cota da API falharem.

## Arquitetura

| Etapa | Arquivo | O que faz |
| --- | --- | --- |
| Entrada | `app.py`, `src/perfil.py` | Formulário guiado (Streamlit), habilidade favorita e montagem do perfil |
| Anonimização | `src/anonimizacao.py` | Troca nome, e-mail, telefone, CPF, CEP e links por marcadores |
| Preparação | `src/llm.py` | LLM converte o perfil em texto de currículo em inglês e neutro em gênero |
| Classificação | `src/classificador.py` | Modelo de ML prevê as 3 áreas mais prováveis (com correção de viés, ver abaixo) e as palavras que mais pesaram |
| Orientação | `src/llm.py`, `src/prompts.py` | LLM gera explicação, feedback, próximos passos e currículo |
| Saída | `app.py`, `src/pdf_curriculo.py` | Relatório, currículo com dados restaurados localmente e download em PDF ou Markdown |
| Visual | `assets/estilo.css`, `.streamlit/config.toml` | Layout de site (hero, como funciona, FAQ, rodapé) |

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

| Modelo | Acurácia | F1 macro |
| --- | --- | --- |
| Regressão Logística | 69,0% | 0,644 |
| **Random Forest** (escolhido) | **74,5%** | **0,685** |

### Do modelo para o app: perfis curtos

O modelo foi treinado com currículos completos, mas no app recebe um perfil curto. Nos
testes, isso causou dois problemas: o Random Forest "chutava" **Desenvolvimento de
Negócios** (cerca de 57%) quando tinha pouca informação, e uma única palavra muito
exclusiva (como "kitchen") vencia várias habilidades de outra área. Por isso, o app:

1. Usa, para cada habilidade do formulário, o **vocabulário real** dos currículos da área
   (`HABILIDADES` em `src/perfil.py`). Cada habilidade, testada sozinha, leva à área
   esperada em 1º lugar (20 de 20; antes eram 5 de 20).
2. Calcula a **média ponderada** entre o perfil completo e cada habilidade separada. A
   habilidade favorita tem peso 3.
3. **Desconta o viés de base**: as probabilidades que o modelo dá sem nenhuma informação.
4. No modo com LLM, junta a tradução feita pela LLM com o vocabulário fixo, para o
   resultado não depender das palavras que a LLM escolheu.

Por causa desses ajustes, as métricas da tabela acima medem o modelo em currículos
completos, **não** o desempenho do app.

## Teste de viés (`bias_test.py`)

Testes contrafactuais: o mesmo currículo, mudando **só** um atributo sensível.

| Teste | Previsões que mudaram |
| --- | --- |
| 1. Gênero, modelo sem mitigação | 2,4% |
| 1. Gênero, modelo com neutralização de gênero | **0,0%** |
| 2. Origem social (escola pública/baixa renda vs. particular/bairro nobre) | 2,4% |
| 3. Modelo do app (Random Forest): gênero / origem social | **0,0%** / 1,6% |
| 4. App de ponta a ponta, 210 perfis do formulário (frase em inglês) | 0,5% |

Resultado completo em `reports/vies.json`. A análise está em [ETICA.md](ETICA.md).

## Ética e privacidade

Resumo do que está implementado. A justificativa completa está em [ETICA.md](ETICA.md).

- **Privacidade (LGPD):** nome, e-mail, telefone e cidade nunca vão para a LLM. O texto
  livre é anonimizado por expressões regulares, e os dados só voltam ao currículo no
  final, localmente. Nada é salvo em disco.
- **Segurança:** a chave de API fica no `.env` (no `.gitignore`). Se a IA falhar, o app usa
  o modo offline, e o texto gerado pela IA nunca é renderizado como HTML.
- **Transparência:** aviso de uso de IA, top 3 áreas com porcentagem, palavras que mais
  pesaram, aba mostrando exatamente o que foi enviado para a IA, e FAQ explicando o uso
  do Gemini.
- **Vieses:** neutralização de gênero, testes contrafactuais, correção do viés de base do
  modelo e prompt que obriga a explicação a seguir a área escolhida pelo modelo.
- **Impacto no usuário:** tom encorajador, valorização de trabalho informal, proibição de
  inventar fatos no currículo, sugestão apresentada como ponto de partida.
- **Limitações:** dataset americano e em inglês; métricas medidas em currículos
  completos; a LLM pode errar; a anonimização não reconhece nomes de terceiros; no plano
  gratuito do Gemini, o Google pode usar os dados recebidos.

## Provedores de LLM

Configure em `.env` com `LLM_PROVIDER`:

| Provedor | Variável da chave | Modelo padrão |
| --- | --- | --- |
| `gemini` (padrão, tem camada gratuita) | `GEMINI_API_KEY` | `gemini-3.5-flash-lite` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |

Para usar outro modelo, defina `LLM_MODEL`. Para OpenAI ou Anthropic, instale também
`openai` ou `anthropic` (estão comentados no `requirements.txt`).
