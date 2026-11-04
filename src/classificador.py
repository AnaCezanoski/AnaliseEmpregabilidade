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
        self._prob_base = None

    def _base(self) -> np.ndarray:
        """Probabilidades que o modelo dá quando não recebe nenhuma informação útil.

        Com textos curtos, o Random Forest "cai" em Desenvolvimento de Negócios
        (cerca de 57%) mesmo sem nenhuma pista dessa área. Medimos esse viés
        uma vez para descontá-lo das previsões.
        """
        if self._prob_base is None:
            neutros = ["", "teamwork", "xyz"]
            self._prob_base = self.pipeline.predict_proba(neutros).mean(axis=0)
        return self._prob_base

    def prever(self, texto_ingles: str, top_k: int = 3,
               partes: list[tuple[str, float]] | None = None) -> list[dict]:
        """Devolve as top_k áreas mais prováveis, com probabilidade.

        `partes` são trechos avaliados separadamente, com peso (ex.: cada
        habilidade marcada, com peso maior para a favorita). O resultado é a
        média ponderada do texto completo e das partes. Assim uma única
        habilidade com palavras muito exclusivas (ex.: "kitchen") não apaga
        várias habilidades de outra área.
        """
        textos = [(texto_ingles, 1.0)] + [(t, w) for t, w in (partes or []) if t.strip()]
        limpos = [limpar_texto(t, self.neutralizar_genero) for t, _ in textos]
        pesos = np.array([w for _, w in textos], dtype=float)
        probs = (self.pipeline.predict_proba(limpos) * pesos[:, None]).sum(axis=0) / pesos.sum()

        ajustadas = np.clip(probs - self._base(), 0, None)
        if ajustadas.sum() > 1e-6:  # se não sobrar sinal nenhum, fica com o original
            probs = ajustadas / ajustadas.sum()

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
