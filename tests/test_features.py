import numpy as np
import pandas as pd

from data.indicators import FEATURE_COLS
from models.features import build_features, build_panel, train_cutoff
from models.train import fit_predict_scores
from tests.conftest import make_prices

HORIZON = 5


def test_label_matches_forward_return():
    prices = make_prices(100, drift=0.01, vol=0.0)  # sobe sempre
    feat = build_features(prices, HORIZON)
    realized = feat["label"].dropna()
    assert (realized == 1.0).all()
    # últimas `horizon` linhas: rótulo ainda não realizado
    assert feat["label"].iloc[-HORIZON:].isna().all()


def test_no_lookahead_in_features():
    """Alterar o futuro não pode mudar features nem rótulos já realizados."""
    prices = make_prices(200)
    base = build_features(prices, HORIZON)

    tampered_prices = prices.copy()
    tampered_prices.iloc[-50:, tampered_prices.columns.get_loc("close")] *= 10
    tampered = build_features(tampered_prices, HORIZON)

    cut = len(prices) - 50 - HORIZON  # linhas cujo rótulo não toca o trecho alterado
    cols = FEATURE_COLS + ["fwd_ret", "label"]
    pd.testing.assert_frame_equal(base[cols].iloc[:cut], tampered[cols].iloc[:cut])


def test_train_cutoff_excludes_unrealized_labels():
    dates = list(range(100))
    i = 60
    cutoff = train_cutoff(dates, i, HORIZON)
    # toda linha de treino (date < cutoff) tem rótulo realizado antes de dates[i]
    train_positions = [p for p in dates if p < cutoff]
    assert all(p + HORIZON < i for p in train_positions)


def test_build_panel_keeps_scoring_rows():
    panel = build_panel({"A": make_prices(100), "B": make_prices(100, seed=1)}, HORIZON)
    assert set(panel["ticker"]) == {"A", "B"}
    assert panel[FEATURE_COLS].notna().all().all()
    last_date = panel["date"].max()
    assert panel[panel["date"] == last_date]["label"].isna().all()


def test_fit_predict_scores_single_class_is_neutral():
    x = pd.DataFrame({"f": [1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 1.0, 1.0])
    scores = fit_predict_scores("gbdt", x, y, x)
    assert np.allclose(scores, 0.5)


def test_fit_predict_scores_learns_separable_pattern():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"f": rng.normal(0, 1, 500)})
    y = (x["f"] > 0).astype(float)
    scores = fit_predict_scores("logistic", x, y, pd.DataFrame({"f": [-3.0, 3.0]}))
    assert scores[0] < 0.2 and scores[1] > 0.8
