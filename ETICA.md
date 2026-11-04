# Aspectos éticos e responsabilidade no uso da IA

Projeto: **Análise de Empregabilidade e Currículo – Assistente Inteligente para Jovens**
(TDE2 de Sistemas de Informação Inteligentes, PUCPR).
Integrantes: Ana Julia Cezanoski, Eduardo Nicolosi e Francielly Pazello.

O sistema orienta jovens, muitos em situação de vulnerabilidade e buscando o primeiro
emprego, sobre áreas profissionais e gera um rascunho de currículo. Uma sugestão errada ou
enviesada pode desanimar alguém ou reforçar desigualdades. Por isso, tratamos a ética como
um requisito do sistema, não como um texto à parte. Cada seção abaixo explica **o risco**,
**o que fizemos** e **por quê**.

> Nome da interface: "Primeiro Passo" é uma marca fictícia, criada só para a apresentação.

---

## 1. Privacidade (LGPD)

**Risco.** Currículos têm dados pessoais (nome, contatos, cidade) e, às vezes, sensíveis.
Mandar esses dados para uma API externa de IA expõe o jovem sem necessidade.

**O que fizemos.**
- **Minimização:** o formulário pede o mínimo. Nome, e-mail, telefone e cidade são
  **opcionais** e ficam em uma seção separada, que avisa que esses dados não vão para a IA.
- **Anonimização antes de qualquer envio** (`src/anonimizacao.py`): a IA recebe marcadores
  como `[NOME]` e `[EMAIL]`. Nos campos de texto livre, expressões regulares também trocam
  e-mail, CPF, CEP, telefone, links e o nome informado pelo jovem.
- **Restauração local:** os dados reais só voltam ao currículo no final, dentro do app,
  depois da resposta da IA (`restaurar()`).
- **Sem armazenamento:** o app não tem banco de dados e não grava arquivos nem cookies
  com dados dos usuários. Tudo fica na sessão e some ao fechar a página. O único cookie é
  o técnico de segurança do Streamlit (proteção XSRF), que não guarda dados pessoais.

**Por quê.** A LGPD (art. 6º) pede finalidade, necessidade e segurança. A IA não precisa
saber *quem* é a pessoa para orientar sobre *o que* ela sabe fazer. Então o dado
identificável simplesmente não sai do app.

## 2. Segurança dos dados

- **Chave de API fora do código:** a chave do Gemini fica no arquivo `.env`, que está no
  `.gitignore` e nunca vai para o GitHub. O repositório tem só um `.env.example` vazio.
- **Dataset fora do repositório:** os currículos do Kaggle (`data/*.csv`) também estão no
  `.gitignore`. O repositório guarda só o modelo treinado.
- **Falha segura:** se a IA não responder (sem internet, cota esgotada, erro), o app usa
  o **modo offline**, com feedback gerado por regras. Assim ele nunca fica sem resposta.
  As chamadas à IA têm tempo limite de 30 s.
- **Saída da IA tratada como dado não confiável:** o texto gerado pela IA é exibido como
  texto, não como HTML. Isso evita que uma resposta maliciosa ou com defeito execute código
  na página.

## 3. Transparência e explicabilidade

- **Aviso de IA** no topo da página e no rodapé: "Esta análise usa inteligência
  artificial e pode errar. Use o resultado como ponto de partida, não como decisão final."
- **Top 3 áreas com porcentagem**, e não uma resposta única. A frase abaixo das barras
  explica que o número mede **semelhança com currículos do dataset**, não a capacidade da
  pessoa.
- **"Por que essa área":** a IA explica o resultado, e o prompt a obriga a falar da área
  que o modelo realmente escolheu. Se o perfil mistura áreas diferentes, ela deve dizer
  isso com honestidade.
- **Palavras que mais pesaram:** o app mostra os termos que mais influenciaram o modelo.
- **Aba "Transparência":** mostra **exatamente** o que foi enviado para a IA (o perfil
  anonimizado), o texto que o classificador recebeu e quais modelos foram usados.
- **FAQ honesta:** a página informa que o perfil anonimizado vai para o Google Gemini e
  que, no plano gratuito, o Google pode usar esses dados.

## 4. Vieses algorítmicos

O dataset (Kaggle, cerca de 2.400 currículos dos EUA) carrega os vieses do mercado de
onde veio. Medimos esses vieses com **testes contrafactuais** (`bias_test.py`): pegamos o
mesmo currículo, mudamos **só** um atributo sensível e verificamos se a sugestão muda. Um
sistema justo deveria dar a mesma resposta nas duas versões.

### Resultados (`reports/vies.json`)

| Teste | O que muda | Previsões que mudaram | Variação média da probabilidade |
|---|---|---|---|
| 1. Gênero, modelo **sem** mitigação | he↔she, man↔woman… (168 currículos) | **2,4%** | 0,4 p.p. |
| 1. Gênero, modelo **com** neutralização | idem | **0,0%** | 0,0 p.p. |
| 2. Origem social (modelo de teste) | frase "escola pública / baixa renda" vs. "escola particular / bairro nobre" (497 currículos) | 2,4% | 1,0 p.p. |
| 3. Gênero, **modelo do app** (Random Forest) | idem teste 1 | **0,0%** | 0,0 p.p. |
| 3. Origem social, **modelo do app** | idem teste 2 | 1,6% | 0,9 p.p. |
| 4. App de ponta a ponta, frase em português | 210 perfis do formulário | 0,0% | 0,1 p.p. |
| 4. App de ponta a ponta, frase em inglês | 210 perfis do formulário | 0,5% (1 perfil) | 3,0 p.p. (máx. 10,9) |

### Mitigações e o que os números mostram

- **Gênero:** sem mitigação, trocar só as palavras de gênero mudava a área sugerida em
  2,4% dos currículos. Nós **removemos os termos de gênero** no treino e na previsão
  (`src/preprocessamento.py`), e o prompt de tradução proíbe pronomes com gênero. Com
  isso, a mudança caiu para **0%**, praticamente sem perder desempenho (F1 de 0,641 para
  0,644).
- **Origem social:** não removemos esse sinal, porque ele aparece de forma indireta
  (escola, bairro, transporte) e não tem como listar todas as palavras. O efeito medido é
  **pequeno, mas não é zero**: 1,6% no modelo do app e 0,5% no app completo. Registramos
  isso como limitação conhecida.
- **Por que o teste 4 roda em dois idiomas:** o modelo só entende inglês. Em português, a
  frase de origem é praticamente ignorada, e o 0% seria **otimista demais**. Em inglês,
  que é o que chega ao modelo quando a IA traduz o perfil, o efeito aparece (3 p.p. em
  média). Preferimos reportar o cenário mais exigente.

### Um viés que encontramos testando o app

Nos testes de uso, um perfil com 3 habilidades de informática e 1 de cozinha recebeu
"Gastronomia" como área principal. Investigando, descobrimos dois problemas:

1. Quando recebe pouca informação, o Random Forest **"chutava" Desenvolvimento de
   Negócios com 57%** de probabilidade, mesmo sem nenhuma pista dessa área. **15 das 20
   habilidades** do formulário, testadas sozinhas, caíam nessa área.
2. Palavras muito exclusivas de uma área (como "kitchen") pesavam mais que várias
   habilidades de outra área.

**Correções** (`src/classificador.py`, `src/perfil.py`):
- Trocamos as palavras de cada habilidade pelo vocabulário real dos currículos de cada
  área, evitando termos ambíguos. "Development", por exemplo, levava "Programação" para
  Desenvolvimento de Negócios.
- O resultado passou a ser a **média do perfil completo com cada habilidade separada**,
  para cada habilidade ter peso e nenhuma palavra isolada dominar.
- Medimos as probabilidades que o modelo dá **sem informação nenhuma** e **descontamos
  esse viés de base**.
- O jovem pode indicar a habilidade de que **mais gosta**, que ganha peso 3.

Resultado: as **20 de 20** habilidades levam à área esperada em 1º lugar, e o perfil do
exemplo passou a receber Tecnologia da Informação. Esse caso mostra por que testar com
**perfis reais do público-alvo** importa: a métrica de teste (74% de acurácia) não
mostrava o problema.

## 5. Impacto sobre os usuários

- **Público vulnerável:** o prompt pede tom **encorajador e honesto** e proíbe ser duro
  ou desmotivador. Ele também pede para valorizar trabalho informal, bico, voluntariado e
  negócio da família, que muitas vezes são a única experiência desse público.
- **Autonomia:** a sugestão é apresentada como **ponto de partida**. A FAQ diz isso
  explicitamente: "A IA decide qual profissão eu devo seguir? Não… A decisão é sempre
  sua." O app mostra 3 áreas, não uma resposta única.
- **Não inventar:** o prompt proíbe criar fatos sobre o jovem (empresas, cursos,
  escolaridade). Isso importa porque um currículo com informação falsa prejudica quem o
  envia. O app pede para revisar o rascunho antes de enviar.
- **Recomendações seguras:** a IA só pode citar instituições conhecidas que oferecem
  cursos gratuitos (SENAI, SENAC, Sebrae, Fundação Bradesco). Ela não pode inventar links,
  valores ou datas.
- **Acesso:** o app é gratuito, não exige cadastro, o formulário leva cerca de 2 minutos e
  funciona mesmo sem a IA (modo offline).

## 6. Limitações da tecnologia

- **Dataset estrangeiro:** os currículos são dos EUA e em inglês. Eles não representam bem
  o mercado brasileiro nem ocupações comuns de primeiro emprego (por exemplo, auxiliar de
  produção, repositor, entregador). Algumas áreas têm poucos exemplos (Automotivo: 36;
  BPO: 22), e as previsões nelas são menos confiáveis.
- **Acurácia medida em outro cenário:** os 74,5% de acurácia (F1 macro de 0,685) foram
  medidos com **currículos completos**. No app, o modelo recebe um perfil curto e passa
  pelos ajustes da seção 4. Essas métricas **não medem diretamente** o desempenho do app.
- **Ajustes feitos à mão:** as palavras de cada habilidade e a correção do viés de base
  foram calibradas pelo grupo. Elas resolvem o problema observado, mas não substituem
  treinar um modelo com dados do público-alvo.
- **A LLM pode errar ou inventar**, mesmo com as regras do prompt. Por isso o app mostra
  avisos e pede para revisar o currículo.
- **Anonimização por regras:** ela pega padrões conhecidos (e-mail, CPF, telefone, o nome
  informado), mas **não reconhece** nomes de terceiros ou de empresas escritos no texto
  livre (por exemplo, "meu tio João"). A FAQ orienta a evitar isso.
- **Dependência de um provedor externo:** no plano gratuito do Gemini, o Google pode
  usar os dados recebidos para melhorar os produtos dele. Mesmo anonimizados, os dados
  saem do nosso controle. Em uso real, o certo seria usar um plano pago com cláusula de
  não uso dos dados ou um modelo aberto rodando localmente. O projeto já aceita trocar o
  provedor (`LLM_PROVIDER`).
- **Testes de viés limitados:** testamos gênero e origem social com frases fixas. Outros
  atributos (raça, idade, deficiência, região) não foram testados, e o resultado depende
  das frases escolhidas.

## Como reproduzir

```bash
python train.py --baixar   # baixa o dataset e treina (gera reports/metricas.json)
python bias_test.py        # testes de viés (gera reports/vies.json)
```
