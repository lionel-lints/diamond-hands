import pandas as pd
import pytest

import cma


def test_cma_file_is_well_formed():
    d = cma.load()
    assert d.return_pct.between(-15, 20).all()
    assert d.source.str.startswith("http").all()
    assert d.groupby(["firm", "asset_class"]).size().max() == 1  # one value per firm x class
    assert d.firm.nunique() >= 10
    gmo = d[d.firm == "GMO"]
    assert gmo.basis.str.contains("real").all()  # converted, and flagged as such


def test_spread_is_per_firm_difference():
    w = pd.DataFrame({"us_large": [5.0, 7.0], "intl_dev": [6.0, None]}, index=["A", "B"])
    s = cma.spread(w)
    assert s.to_dict() == {"A": pytest.approx(1.0)}  # B dropped: no intl forecast


def test_unknown_class_rejected(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    pd.DataFrame([{"firm": "X", "asset_class": "gold", "return_pct": 1, "source": "http://x"}]).to_csv(tmp_path / "data/cma.csv", index=False)
    monkeypatch.setattr(cma, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="gold"):
        cma.load()
