"""Treina e compara os modelos de classificação de área profissional.

Uso:
    python train.py                      # usa data/Resume.csv
    python train.py --dados outro.csv    # outro arquivo (ou vários)
    python train.py --baixar             # baixa o dataset do Kaggle via kagglehub

Saídas:
    models/modelo.joblib                 melhor modelo (usado pelo app)
    reports/metricas.json                métricas dos dois modelos
    reports/relatorio_<modelo>.txt       precisão, recall e F1 por área
    reports/matriz_confusao_<modelo>.png matriz de confusão normalizada
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.categorias import nome_pt
from src.preprocessamento import limpar_texto

RAIZ = Path(__file__).parent
SEED = 42

# Nomes de colunas dos dois datasets de currículos mais usados no Kaggle.
COLUNAS_TEXTO = ["Resume_str", "Resume", "resume_text", "text"]
COLUNAS_ROTULO = ["Category", "category", "label"]


def carregar_dados(caminhos: list[Path]) -> pd.DataFrame:
    partes = []
    for caminho in caminhos:
        df = pd.read_csv(caminho)
        col_texto = next((c for c in COLUNAS_TEXTO if c in df.columns), None)
        col_rotulo = next((c for c in COLUNAS_ROTULO if c in df.columns), None)
        if not col_texto or not col_rotulo:
            raise ValueError(
                f"{caminho}: não encontrei colunas de texto/rótulo. Colunas: {list(df.columns)}"
            )
        partes.append(
            pd.DataFrame({"texto": df[col_texto], "area": df[col_rotulo].str.strip().str.upper()})
        )
    dados = pd.concat(partes, ignore_index=True).dropna()
    return dados.drop_duplicates(subset="texto")


def baixar_kaggle() -> Path:
    import kagglehub

    pasta = Path(kagglehub.dataset_download("snehaanbhawal/resume-dataset"))
    candidatos = list(pasta.rglob("Resume.csv"))
    if not candidatos:
        raise FileNotFoundError(f"Resume.csv não encontrado em {pasta}")
    return candidatos[0]


def montar_modelos() -> dict[str, Pipeline]:
    def tfidf():
        return TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )

    return {
        "regressao_logistica": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(max_iter=3000, C=10, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("tfidf", tfidf()),
            ("clf", RandomForestClassifier(
                n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=SEED
            )),
        ]),
    }


def salvar_matriz_confusao(y_true, y_pred, classes, nome, destino: Path):
    matriz = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    rotulos = [nome_pt(c) for c in classes]
    fig, ax = plt.subplots(figsize=(13, 11), dpi=130)
    im = ax.imshow(matriz, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), rotulos, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), rotulos, fontsize=8)
    ax.set_xlabel("Área prevista")
    ax.set_ylabel("Área real")
    ax.set_title(f"Matriz de confusão normalizada – {nome.replace('_', ' ').title()}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if matriz[i, j] >= 0.05:
                ax.text(j, i, f"{matriz[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if matriz[i, j] > 0.5 else "#222")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dados", nargs="+", type=Path, default=[RAIZ / "data" / "Resume.csv"])
    parser.add_argument("--baixar", action="store_true", help="baixa o dataset do Kaggle")
    args = parser.parse_args()

    caminhos = [baixar_kaggle()] if args.baixar else args.dados
    dados = carregar_dados(caminhos)
    print(f"{len(dados)} currículos, {dados['area'].nunique()} áreas")
    print(dados["area"].value_counts().to_string(), "\n")

    print("Pré-processando textos...")
    X = dados["texto"].map(limpar_texto)
    y = dados["area"]
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    (RAIZ / "reports").mkdir(exist_ok=True)
    (RAIZ / "models").mkdir(exist_ok=True)
    classes = sorted(y.unique())
    metricas, treinados = {}, {}

    for nome, modelo in montar_modelos().items():
        inicio = time.time()
        modelo.fit(X_treino, y_treino)
        pred = modelo.predict(X_teste)
        metricas[nome] = {
            "acuracia": round(accuracy_score(y_teste, pred), 4),
            "precisao_macro": round(precision_score(y_teste, pred, average="macro", zero_division=0), 4),
            "recall_macro": round(recall_score(y_teste, pred, average="macro", zero_division=0), 4),
            "f1_macro": round(f1_score(y_teste, pred, average="macro", zero_division=0), 4),
            "f1_ponderado": round(f1_score(y_teste, pred, average="weighted", zero_division=0), 4),
            "tempo_treino_s": round(time.time() - inicio, 1),
        }
        treinados[nome] = modelo
        relatorio = classification_report(y_teste, pred, zero_division=0)
        (RAIZ / "reports" / f"relatorio_{nome}.txt").write_text(relatorio, encoding="utf-8")
        salvar_matriz_confusao(
            y_teste, pred, classes, nome, RAIZ / "reports" / f"matriz_confusao_{nome}.png"
        )
        print(f"== {nome} ==")
        print(json.dumps(metricas[nome], indent=2))
        print(relatorio)

    # O critério de escolha é o F1 macro: trata todas as áreas com o mesmo peso,
    # o que importa porque o dataset é desbalanceado.
    melhor = max(metricas, key=lambda n: metricas[n]["f1_macro"])
    joblib.dump(
        {
            "pipeline": treinados[melhor],
            "nome": melhor,
            "classes": list(treinados[melhor].classes_),
            "neutralizar_genero": True,
            "metricas": metricas[melhor],
        },
        RAIZ / "models" / "modelo.joblib",
    )
    resumo = {
        "melhor_modelo": melhor,
        "criterio": "f1_macro",
        "n_curriculos": int(len(dados)),
        "n_areas": int(len(classes)),
        "n_teste": int(len(y_teste)),
        "distribuicao_areas": {k: int(v) for k, v in y.value_counts().items()},
        "modelos": metricas,
    }
    (RAIZ / "reports" / "metricas.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Melhor modelo: {melhor} (F1 macro = {metricas[melhor]['f1_macro']})")
    print("Salvo em models/modelo.joblib")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
