"""Total return of US size/style funds, international and bonds over past crash windows.
Reads data/prices/style_monthly.csv (Vanguard index mutual funds, month-end adjusted close).
Run: uv run --with pandas python scripts/style_windows.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WINDOWS = {
    "Dot-com 2000-03→2002-09": ("2000-03", "2002-09"),
    "GFC 2007-10→2009-02": ("2007-10", "2009-02"),
    "COVID 2020-01→2020-03": ("2020-01", "2020-03"),
    "2022 inflation": ("2021-12", "2022-12"),
    "AI run 2023-01→now": ("2022-12", None),
}


def window_returns(d):
    """Start-to-end total return per column for each window; None end means latest month."""
    return pd.DataFrame({n: d.loc[b or d.index[-1]] / d.loc[a] - 1 for n, (a, b) in WINDOWS.items()}).T


def main():
    d = pd.read_csv(ROOT / "data/prices/style_monthly.csv", index_col=0)
    print((window_returns(d) * 100).round(0).astype("Int64").to_string())


if __name__ == "__main__":
    main()
