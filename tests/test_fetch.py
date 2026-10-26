import pytest

import fetch_fund_holdings as FH


class FakeResp:
    def __init__(self, text):
        self.text = text
        self.content = text.encode()

    def raise_for_status(self):
        pass


@pytest.fixture
def fake_out(tmp_path, monkeypatch):
    monkeypatch.setattr(FH, "ROOT", tmp_path)
    monkeypatch.setattr(FH, "OUT", tmp_path / "data/fund_holdings")
    monkeypatch.setattr(FH, "ISHARES", {"ITOT": "1/x"})
    return tmp_path


def test_saves_csv_and_uses_latest_holdings_url(fake_out, monkeypatch):
    urls = []
    monkeypatch.setattr(FH.requests, "get", lambda url, **kw: urls.append(url) or FakeResp("iShares Fake ETF\nTicker,Name\n"))
    FH.main()
    assert urls == ["https://www.ishares.com/us/products/1/x/latest-holdings.csv"]
    files = list((fake_out / "data/fund_holdings").glob("ITOT_*.csv"))
    assert len(files) == 1 and files[0].read_text().startswith("iShares")


def test_rejects_html_instead_of_csv(fake_out, monkeypatch):
    monkeypatch.setattr(FH.requests, "get", lambda url, **kw: FakeResp("<!DOCTYPE html><html>"))
    with pytest.raises(RuntimeError, match="got HTML"):
        FH.main()
    assert not list((fake_out / "data/fund_holdings").glob("*.csv"))
