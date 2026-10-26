"""Download published holdings for proxy ETFs into data/fund_holdings/.

Vanguard blocks scripted downloads, so each model block is mapped to an iShares ETF
tracking the same or a near-identical index (see PROXIES in lookthrough.py).
Run: uv run --with requests python scripts/fetch_fund_holdings.py
"""
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/fund_holdings"

ISHARES = {
    "ITOT": "239724/ishares-core-s-p-total-u-s-stock-market-etf",
    "IVV": "239726/ishares-core-s-p-500-etf",
    "IXUS": "244048/ishares-core-msci-total-international-stock-etf",
    "ISCV": "239588/ishares-morningstar-small-cap-value-etf",
    "ISCG": "239587/ishares-morningstar-small-cap-growth-etf",
    # 401(k) menu proxies
    "IMCB": "239582/ishares-morningstar-mid-cap-etf",  # Vanguard Morningstar Mid-Cap Index
    "EFA": "239623/ishares-msci-eafe-etf",  # iShares MSCI EAFE Intl Index K (same index)
    "IEMG": "244050/ishares-core-msci-emerging-markets-etf",  # Vanguard EM (FTSE; Korea dropped in lookthrough)
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for tk, path in ISHARES.items():
        url = f"https://www.ishares.com/us/products/{path}/latest-holdings.csv"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        r.raise_for_status()
        if not r.text.lstrip().startswith("iShares"):
            raise RuntimeError(f"{tk}: got HTML instead of holdings CSV from {url}")
        f = OUT / f"{tk}_{date.today():%Y-%m-%d}.csv"
        f.write_bytes(r.content)
        print(f"{tk}: {len(r.content):,} bytes -> {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
