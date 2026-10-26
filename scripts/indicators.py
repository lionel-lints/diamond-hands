"""Review indicators (IPS sec. 9): is the market concentration that justifies the profile's hedges still there?

Four measurements, each compared with the baseline and the "supports shrinking the hedges"
threshold in the profile's [review_indicators]:
  cape                 US Shiller CAPE (multpl.com, monthly; cached in data/prices/cape.csv)
  sp500_top10          top-10 issuers' share of the S&P 500 (IVV holdings; GOOG+GOOGL merged)
  core_ai_us_market    core AI basket's share of the US total market (ITOT holdings)
  intl_favored_share   share of firms in data/cma.csv forecasting intl dev above US large

Holdings-based indicators are only as fresh as data/fund_holdings/ (fetch_fund_holdings.py).
The indicators inform the scheduled review; they do not trigger trades.

Run: uv run --with pandas --with requests python scripts/indicators.py [--profile NAME] [--offline]
"""
import argparse
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

import cma
import lookthrough as L
import user_profile as P

ROOT = Path(__file__).resolve().parent.parent
CAPE_URL = "https://www.multpl.com/shiller-pe/table/by-month"
CAPE_CACHE = ROOT / "data/prices/cape.csv"


def parse_cape_table(html):
    """[(date string, CAPE)] from multpl's by-month table, newest first."""
    flat = re.sub(r"\s+", " ", html)
    rows = re.findall(r"<td[^>]*>\s*([A-Z][a-z]{2} \d{1,2}, \d{4})\s*</td>\s*<td[^>]*>\s*(?:&#x2002;)?\s*([\d.]+)", flat)
    return [(d, float(v)) for d, v in rows]


def fetch_cape():
    import requests
    html = requests.get(CAPE_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60).text
    rows = parse_cape_table(html)
    if not rows:
        raise RuntimeError(f"no CAPE rows parsed from {CAPE_URL}")
    df = pd.DataFrame(rows, columns=["date", "cape"])
    df["date"] = pd.to_datetime(df.date).dt.date
    df.sort_values("date").to_csv(CAPE_CACHE, index=False)
    return df


def latest_cape(offline=False):
    if not offline:
        try:
            fetch_cape()
        except Exception as e:  # network down: fall back to the cache, and say so
            print(f"  (CAPE fetch failed: {e}; using cache)")
    d = pd.read_csv(CAPE_CACHE)
    return float(d.cape.iloc[-1]), str(d.date.iloc[-1])


def top10_share(etf="IVV"):
    h = L.tag(L.load_fund(etf).assign(dollars=lambda d: d.w, account="", source=etf, sleeve="equity"))
    return h.groupby("issuer").dollars.sum().nlargest(10).sum()


def core_ai_share(etf="ITOT"):
    h = L.tag(L.load_fund(etf).assign(dollars=lambda d: d.w, account="", source=etf, sleeve="equity"))
    return h[h.ai_tier == "core"].dollars.sum()


def intl_favored_share():
    s = cma.spread(cma.wide())
    return (s > 0).mean(), int((s > 0).sum()), len(s)


def measure(offline=False):
    cape, cape_date = latest_cape(offline)
    fav, n_fav, n = intl_favored_share()
    holdings_date = max(f.stem.split("_")[-1] for f in (ROOT / "data/fund_holdings").glob("IVV_*.csv"))
    return {
        "cape": (cape, f"multpl.com, {cape_date}"),
        "sp500_top10": (top10_share(), f"IVV holdings {holdings_date}"),
        "core_ai_us_market": (core_ai_share(), f"ITOT holdings {holdings_date}"),
        "intl_favored_share": (fav, f"{n_fav} of {n} firms (data/cma.csv)"),
    }


def status(value, cfg):
    """'hedge still justified' unless the value has crossed the shrink threshold."""
    return "SUPPORTS SHRINKING" if value <= cfg["shrink_below"] else "concentration persists"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile")
    p.add_argument("--offline", action="store_true", help="use the cached CAPE")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    P.use(args.profile) if args.profile else P.current()
    profile = P.current()
    cfg = profile.settings.get("review_indicators", {})
    print(profile.summary(), "\n")
    print(f"Review indicators as of {date.today()} (IPS sec. 9; next scheduled review: "
          f"{profile.settings.get('review_by', 'n/a')})\n")
    rows = []
    for key, (value, src) in measure(args.offline).items():
        c = cfg.get(key)
        fmt = "{:.1f}" if key == "cape" else "{:.0%}"
        line = {"indicator": key, "now": fmt.format(value), "source": src}
        if c:
            line.update({"baseline": fmt.format(c["baseline"]), "shrink below": fmt.format(c["shrink_below"]),
                         "status": status(value, c)})
        rows.append(line)
    t = pd.DataFrame(rows).set_index("indicator")
    print(t.to_string())
    n = (t.get("status") == "SUPPORTS SHRINKING").sum() if "status" in t else 0
    print(f"\n{n} of {len(t)} indicators support shrinking the hedges. These inform the scheduled review; "
          "they don't trigger trades.")


if __name__ == "__main__":
    main()
