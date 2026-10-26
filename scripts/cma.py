"""Capital market assumptions from many firms (data/cma.csv), so no single forecast drives a
result.

Every row is a firm x asset class annualized NOMINAL return in USD, with its basis, horizon,
publication date and source. Most rows come from a third-party compilation (PortfolioLab),
cross-checked against primary sources for Vanguard, J.P. Morgan, AQR, BlackRock, GMO;
Research Affiliates' bond figure differs between sources (4.7% vs 5.1%). GMO publishes
7-year REAL returns; they are converted with an assumed 2.5% inflation (flagged in `basis`).
Run: uv run --with pandas python scripts/cma.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["us_large", "us_small", "us_small_value", "intl_dev", "em", "us_bonds", "tips", "cash"]


def load():
    d = pd.read_csv(ROOT / "data/cma.csv")
    bad = set(d.asset_class) - set(CLASSES)
    if bad:
        raise ValueError(f"data/cma.csv: unknown asset classes {sorted(bad)}")
    return d


def wide(d=None):
    d = load() if d is None else d
    return d.pivot_table(index="firm", columns="asset_class", values="return_pct").reindex(columns=CLASSES)


def spread(w, a="intl_dev", b="us_large"):
    """Per-firm forecast advantage of `a` over `b` (pp per year)."""
    return (w[a] - w[b]).dropna().sort_values()


def main():
    w = wide()
    print("10-yr nominal forecasts by firm (GMO: 7-yr real + 2.5%):\n")
    print(w.dropna(axis=1, how="all").round(1).to_string(na_rep="—"), "\n")
    print("Median / min / max across firms:")
    print(w.agg(["median", "min", "max", "count"]).T.dropna().round(1).to_string(), "\n")
    s = spread(w)
    print(f"International developed minus US large, per firm (pp/yr): median {s.median():+.1f}, "
          f"{(s > 0).sum()} of {len(s)} firms favor international")
    print("  " + ", ".join(f"{f} {v:+.1f}" for f, v in s.items()))


if __name__ == "__main__":
    main()
