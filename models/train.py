"""Modelos de classificação de tendência: baseline logístico vs. gradient boosting."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["logistic", "gbdt"]


def make_model(name: str):
    if name == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ])
    if name == "gbdt":
        return HistGradientBoostingClassifier(
            max_iter=200, max_depth=3, learning_rate=0.05, random_state=0
        )
    raise ValueError(f"modelo desconhecido: {name!r} (opções: {MODEL_NAMES})")


def fit_predict_scores(
    model_name: str, train_x: pd.DataFrame, train_y: pd.Series, score_x: pd.DataFrame
) -> np.ndarray:
    """Treina do zero e retorna P(alta) para score_x.

    Com uma única classe no treino (ou treino vazio) devolve score neutro 0.5.
    """
    if len(train_x) == 0 or train_y.nunique() < 2:
        return np.full(len(score_x), 0.5)
    model = make_model(model_name)
    model.fit(train_x, train_y)
    return model.predict_proba(score_x)[:, 1]
