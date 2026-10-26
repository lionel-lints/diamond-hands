"""Monthly adjusted closes for every return series the replay uses -> data/prices/replay_monthly.csv.
Run: uv run --with yfinance --with pandas python scripts/fetch_prices.py [--profile NAME]
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
TICKERS = ["VTSMX", "VFINX", "VGTSX", "NAESX", "VISVX", "VIGRX", "VBMFX", "VIPSX", "VGSIX", "ETH-USD",
           # Common plan-menu blocks: mid cap, developed ex-US, EM, intermediate bond, short TIPS
           "VIMSX", "VTMGX", "VEIEX", "VBIIX", "VTIP",
           # Japan 1990- lost decades (price index, no dividends) and Nasdaq-100 (AI-heavy proxy)
           "^N225", "^NDX"]


def profile_tickers():
    """Directly held stocks in the active profile (their own return history for the replay)."""
    import blocks as B
    import lookthrough as L
    import user_profile as P

    P.from_argv(sys.argv[1:])
    return sorted({b for b in L.load_positions().block if B.is_direct_stock(b)} - set(TICKERS))


def main():
    cols = {}
    for t in TICKERS + profile_tickers():
        for attempt in range(3):
            s = yf.Ticker(t).history(start="1965-01-01", interval="1mo", auto_adjust=True)["Close"]
            if len(s):
                break
            time.sleep(2)
        if not len(s):
            print(f"{t:8} NO DATA (skipped)")
            continue
        s.index = s.index.tz_localize(None).to_period("M")
        cols[t] = s[~s.index.duplicated(keep="last")]
        print(f"{t:8} {cols[t].index[0]} → {cols[t].index[-1]}")
    pd.DataFrame(cols).to_csv(ROOT / "data/prices/replay_monthly.csv")


if __name__ == "__main__":
    main()
