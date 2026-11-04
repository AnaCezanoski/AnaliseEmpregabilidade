"""Teste de viés com versões contrafactuais.

Ideia: pegar cada currículo do conjunto de teste, criar uma cópia que muda só
um atributo sensível e ver se a previsão do modelo muda. Um modelo justo
deveria dar a mesma área para as duas versões.

Quatro testes:
  1. Gênero: troca he<->she, his<->her, man<->woman etc.
     Compara um modelo SEM neutralização de gênero com um modelo COM
     neutralização (a mitigação usada no app).
  2. Origem social: acrescenta ao currículo uma frase sobre escola pública em
     comunidade de baixa renda ou sobre escola particular em bairro nobre.
     Mede se esse tipo de informação, que aparece indiretamente nos
     currículos, mexe na sugestão de área.
  3. Modelo do app: repete os testes 1 e 2 com o modelo salvo em
     models/modelo.joblib (o que o app usa de verdade), no mesmo conjunto
     de teste, que ele não viu no treino.
  4. App de ponta a ponta: perfis do formulário com e sem uma frase sobre
     origem social no campo de interesses, passando pelo mesmo caminho do app
     (glossário + média por habilidade + correção do viés de base).

Uso:  python bias_test.py [--dados data/Resume.csv]
Saída: reports/vies.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.classificador import Classificador
from src.perfil import HABILIDADES, Perfil
from src.preprocessamento import limpar_texto
from train import SEED, carregar_dados

RAIZ = Path(__file__).parent

TROCAS_GENERO = {
    "he": "she", "she": "he", "him": "her", "his": "her", "her": "his",
    "hers": "his", "himself": "herself", "herself": "himself",
    "mr": "ms", "ms": "mr", "mrs": "mr", "male": "female", "female": "male",
    "man": "woman", "woman": "man", "men": "women", "women": "men",
    "boy": "girl", "girl": "boy", "husband": "wife", "wife": "husband",
    "father": "mother", "mother": "father", "son": "daughter", "daughter": "son",
    "brother": "sister", "sister": "brother", "fraternity": "sorority",
    "sorority": "fraternity", "paternity": "maternity", "maternity": "paternity",
}
_PALAVRA = re.compile(r"\b(" + "|".join(TROCAS_GENERO) + r")\b", re.IGNORECASE)

FRASES_ORIGEM = {
    "baixa_renda": " Studied at a public school in a low income community. "
                   "Lives in a low income neighborhood and uses public transportation.",
    "alta_renda": " Studied at a private school in an upscale neighborhood. "
                  "Lives in a wealthy area and has a car.",
}


def trocar_genero(texto: str) -> str:
    def troca(m):
        nova = TROCAS_GENERO[m.group(0).lower()]
        return nova.capitalize() if m.group(0)[0].isupper() else nova
    return _PALAVRA.sub(troca, texto)


def treinar(textos, rotulos, neutralizar: bool) -> Pipeline:
    modelo = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=3000, C=10, class_weight="balanced")),
    ])
    modelo.fit([limpar_texto(t, neutralizar) for t in textos], rotulos)
    return modelo


FRASES_ORIGEM_PT = {
    "baixa_renda": "Estudei em escola pública na periferia e moro em comunidade de baixa renda.",
    "alta_renda": "Estudei em escola particular e moro em bairro nobre.",
}


def perfis_de_teste() -> list[Perfil]:
    """Perfis do formulário: cada habilidade sozinha e todos os pares de habilidades."""
    habs = [h for h, termos in HABILIDADES.items() if termos]
    combinacoes = [[h] for h in habs] + [[a, b] for i, a in enumerate(habs) for b in habs[i + 1:]]
    return [Perfil(escolaridade="Ensino médio completo", habilidades=c) for c in combinacoes]


def testar_app(clf: Classificador, frases: dict[str, str]) -> dict:
    """Compara a área sugerida pelo app para o mesmo perfil com origem social diferente.

    Roda com a frase em português (modo offline: o modelo, treinado em inglês,
    quase não reconhece essas palavras) e em inglês (o que chega ao modelo
    quando a LLM traduz o perfil), que é o cenário mais exigente.
    """
    mudou, variacoes = 0, []
    perfis = perfis_de_teste()
    for p in perfis:
        res = {}
        for chave, frase in frases.items():
            p.interesses = frase
            res[chave] = clf.prever(p.texto_ingles_offline(), 1, partes=p.partes_classificador())[0]
        mudou += res["baixa_renda"]["codigo"] != res["alta_renda"]["codigo"]
        variacoes.append(abs(res["baixa_renda"]["probabilidade"] - res["alta_renda"]["probabilidade"]))
    return {
        "n_perfis": len(perfis),
        "taxa_mudanca_previsao": round(mudou / len(perfis), 4),
        "variacao_media_probabilidade": round(float(np.mean(variacoes)), 4),
        "variacao_maxima_probabilidade": round(float(np.max(variacoes)), 4),
    }


def comparar(modelo, neutralizar, textos_a, textos_b) -> dict:
    """Taxa de previsões que mudam entre a versão A e a versão B."""
    xa = [limpar_texto(t, neutralizar) for t in textos_a]
    xb = [limpar_texto(t, neutralizar) for t in textos_b]
    pa, pb = modelo.predict_proba(xa), modelo.predict_proba(xb)
    pred_a, pred_b = pa.argmax(1), pb.argmax(1)
    delta = np.abs(pa[np.arange(len(xa)), pred_a] - pb[np.arange(len(xb)), pred_a])
    return {
        "n_curriculos": len(xa),
        "taxa_mudanca_previsao": round(float((pred_a != pred_b).mean()), 4),
        "variacao_media_probabilidade": round(float(delta.mean()), 4),
        "variacao_maxima_probabilidade": round(float(delta.max()), 4) if len(delta) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dados", nargs="+", type=Path, default=[RAIZ / "data" / "Resume.csv"])
    args = parser.parse_args()

    dados = carregar_dados(args.dados)
    X_tr, X_te, y_tr, y_te = train_test_split(
        dados["texto"], dados["area"], test_size=0.2, stratify=dados["area"], random_state=SEED
    )
    X_te = list(X_te)

    # Só faz sentido trocar gênero em currículos que têm alguma palavra de gênero.
    com_genero = [t for t in X_te if _PALAVRA.search(t)]
    trocados = [trocar_genero(t) for t in com_genero]

    resultado = {"genero": {}, "origem_social": {}}
    for neutralizar, rotulo in [(False, "sem_neutralizacao"), (True, "com_neutralizacao")]:
        modelo = treinar(X_tr, y_tr, neutralizar)
        f1 = f1_score(y_te, modelo.predict([limpar_texto(t, neutralizar) for t in X_te]),
                      average="macro")
        info = {"f1_macro_teste": round(float(f1), 4)}
        if com_genero:
            info.update(comparar(modelo, neutralizar, com_genero, trocados))
        else:
            info["observacao"] = "nenhum currículo do teste tem palavras de gênero"
        resultado["genero"][rotulo] = info

        if neutralizar:  # o teste de origem usa o modelo mitigado (o do app)
            baixa = [t + FRASES_ORIGEM["baixa_renda"] for t in X_te]
            alta = [t + FRASES_ORIGEM["alta_renda"] for t in X_te]
            resultado["origem_social"] = comparar(modelo, neutralizar, baixa, alta)

    # Testes 3 e 4: o modelo e o caminho que o app usa de verdade.
    clf = Classificador()
    modelo_app = {"modelo": clf.nome}
    if com_genero:
        modelo_app["genero"] = comparar(clf.pipeline, clf.neutralizar_genero, com_genero, trocados)
    modelo_app["origem_social"] = comparar(
        clf.pipeline, clf.neutralizar_genero,
        [t + FRASES_ORIGEM["baixa_renda"] for t in X_te],
        [t + FRASES_ORIGEM["alta_renda"] for t in X_te],
    )
    resultado["modelo_do_app"] = modelo_app
    resultado["app_ponta_a_ponta"] = {
        "frase_em_portugues": testar_app(clf, FRASES_ORIGEM_PT),
        "frase_em_ingles": testar_app(clf, {k: v.strip() for k, v in FRASES_ORIGEM.items()}),
    }

    (RAIZ / "reports").mkdir(exist_ok=True)
    (RAIZ / "reports" / "vies.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("TESTE 1 – GÊNERO (currículos com palavras de gênero trocadas)")
    for rotulo, info in resultado["genero"].items():
        print(f"  {rotulo:20s} F1={info['f1_macro_teste']:.3f}  "
              f"mudança de previsão={info.get('taxa_mudanca_previsao', 0):.1%}  "
              f"variação média de prob.={info.get('variacao_media_probabilidade', 0):.3f}")
    o = resultado["origem_social"]
    print("\nTESTE 2 – ORIGEM SOCIAL (frase de baixa renda vs. alta renda)")
    print(f"  mudança de previsão={o['taxa_mudanca_previsao']:.1%}  "
          f"variação média de prob.={o['variacao_media_probabilidade']:.3f}")
    m = resultado["modelo_do_app"]
    print(f"\nTESTE 3 – MODELO DO APP ({m['modelo']})")
    if "genero" in m:
        print(f"  gênero: mudança de previsão={m['genero']['taxa_mudanca_previsao']:.1%}  "
              f"variação média de prob.={m['genero']['variacao_media_probabilidade']:.3f}")
    print(f"  origem social: mudança de previsão={m['origem_social']['taxa_mudanca_previsao']:.1%}  "
          f"variação média de prob.={m['origem_social']['variacao_media_probabilidade']:.3f}")
    print("\nTESTE 4 – APP DE PONTA A PONTA (perfis do formulário, origem social)")
    for idioma, a in resultado["app_ponta_a_ponta"].items():
        print(f"  {idioma:20s} perfis={a['n_perfis']}  mudança de previsão={a['taxa_mudanca_previsao']:.1%}  "
              f"variação média de prob.={a['variacao_media_probabilidade']:.3f}  "
              f"máxima={a['variacao_maxima_probabilidade']:.3f}")
    print("\nResultado salvo em reports/vies.json")


if __name__ == "__main__":
    main()
