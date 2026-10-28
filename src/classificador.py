"""Carrega o modelo treinado e faz a previsão com explicação."""

from pathlib import Path

import joblib
import numpy as np

from src.categorias import nome_pt
from src.preprocessamento import limpar_texto

CAMINHO_MODELO = Path(__file__).resolve().parent.parent / "models" / "modelo.joblib"


class Classificador:
    def __init__(self, caminho: Path = CAMINHO_MODELO):
        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {caminho}. Rode 'python train.py' primeiro."
            )
        pacote = joblib.load(caminho)
        self.pipeline = pacote["pipeline"]
        self.nome = pacote["nome"]
        self.metricas = pacote.get("metricas", {})
        self.neutralizar_genero = pacote.get("neutralizar_genero", True)
        self.tfidf = self.pipeline.named_steps["tfidf"]
        self.clf = self.pipeline.named_steps["clf"]
        self.termos = self.tfidf.get_feature_names_out()

    def prever(self, texto_ingles: str, top_k: int = 3) -> list[dict]:
        """Devolve as top_k áreas mais prováveis, com probabilidade."""
        limpo = limpar_texto(texto_ingles, self.neutralizar_genero)
        probs = self.pipeline.predict_proba([limpo])[0]
        ordem = np.argsort(probs)[::-1][:top_k]
        return [
            {"codigo": self.clf.classes_[i], "nome": nome_pt(self.clf.classes_[i]),
             "probabilidade": float(probs[i])}
            for i in ordem
        ]

    def termos_relevantes(self, texto_ingles: str, codigo_area: str, n: int = 8) -> list[str]:
        """Palavras do texto que mais empurraram o modelo para a área prevista."""
        limpo = limpar_texto(texto_ingles, self.neutralizar_genero)
        vetor = self.tfidf.transform([limpo]).toarray()[0]
        if hasattr(self.clf, "coef_"):  # Regressão Logística: peso da classe
            idx = list(self.clf.classes_).index(codigo_area)
            contribuicao = vetor * self.clf.coef_[idx]
        else:  # Random Forest: importância global da palavra
            contribuicao = vetor * self.clf.feature_importances_
        ordem = np.argsort(contribuicao)[::-1]
        return [self.termos[i] for i in ordem[:n] if contribuicao[i] > 0]
