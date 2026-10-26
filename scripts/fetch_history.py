"""Long-history inputs:

data/prices/ff_monthly.csv   Ken French Data Library, monthly total returns (decimal):
    us_mkt, rf (1926-), us_small_value/us_small_growth/us_big_value/us_big_growth
    (2x3 size x book-to-market, value-weighted, 1926-), dev_mkt, dev_small_value,
    dev_big_growth (developed ex-US, USD, 1990-)
data/prices/ff_japan_monthly.csv  Ken French Japan market / small value / big growth / big
    value (USD, 1990-07-), for the Japan-style scenarios
data/prices/macro_monthly.csv  FRED: cpi (CPIAUCNS, 1913-), gs10 (10-yr Treasury yield, %),
    baa10y (Moody's Baa minus 10-yr Treasury, pp, 1953-), tb3m (3-month T-bill, %),
    t10yie (10-yr breakeven inflation, %, 2003-): the market's expected inflation

Run: uv run --with pandas --with requests python scripts/fetch_history.py
"""
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
KF = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def first_monthly_table(text):
    """Parse the first table in a Ken French CSV: a ',col,col' header followed by YYYYMM rows.
    In these files the first table is the monthly, value-weighted one. Percent -> decimal."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(","))
    cols = [c.strip() for c in lines[start].split(",")[1:]]
    rows = []
    for l in lines[start + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if not (len(parts[0]) == 6 and parts[0].isdigit()):
            break
        rows.append([parts[0]] + [float(p) for p in parts[1:]])
    df = pd.DataFrame(rows, columns=["ym"] + cols).set_index("ym")
    df.index = pd.PeriodIndex([f"{s[:4]}-{s[4:]}" for s in df.index], freq="M")
    return df.where(df > -99) / 100  # -99.99 / -999 mark missing


def kf(name):
    z = zipfile.ZipFile(io.BytesIO(requests.get(KF.format(name), timeout=60).content))
    return first_monthly_table(z.read(z.namelist()[0]).decode("latin-1"))


def fred(series):
    df = pd.read_csv(io.StringIO(requests.get(FRED.format(series), timeout=60).text))
    s = df.set_index(df.columns[0]).iloc[:, 0]
    s.index = pd.PeriodIndex(pd.to_datetime(s.index), freq="M")
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s.groupby(level=0).last().rename(series)  # daily series (e.g. T10YIE) -> month-end


def main():
    fac = kf("F-F_Research_Data_Factors")
    six = kf("6_Portfolios_2x3")
    dev_fac = kf("Developed_ex_US_3_Factors")
    dev_six = kf("Developed_ex_US_6_Portfolios_ME_BE-ME")
    ff = pd.DataFrame({
        "us_mkt": fac["Mkt-RF"] + fac["RF"], "rf": fac["RF"],
        "us_small_value": six["SMALL HiBM"], "us_small_growth": six["SMALL LoBM"],
        "us_big_value": six["BIG HiBM"], "us_big_growth": six["BIG LoBM"],
        "dev_mkt": dev_fac["Mkt-RF"] + dev_fac["RF"],
        "dev_small_value": dev_six["SMALL HiBM"], "dev_big_growth": dev_six["BIG LoBM"],
    })
    macro = pd.DataFrame({"cpi": fred("CPIAUCNS"), "gs10": fred("GS10"),
                          "baa10y": fred("BAA10YM"), "tb3m": fred("TB3MS"), "t10yie": fred("T10YIE")})
    out = ROOT / "data/prices"
    ff.to_csv(out / "ff_monthly.csv")
    # Japan (Bloomberg-based, USD, 1990-07-) for the Japan-style scenarios in data/scenarios.csv
    jf, j6 = kf("Japan_3_Factors"), kf("Japan_6_Portfolios_ME_BE-ME")
    pd.DataFrame({"jp_mkt": jf["Mkt-RF"] + jf["RF"], "jp_small_value": j6["SMALL HiBM"],
                  "jp_big_growth": j6["BIG LoBM"], "jp_big_value": j6["BIG HiBM"]}).to_csv(out / "ff_japan_monthly.csv")
    macro.to_csv(out / "macro_monthly.csv")
    for name, df in [("ff_monthly", ff), ("macro_monthly", macro)]:
        for c in df:
            s = df[c].dropna()
            print(f"{name:14} {c:16} {s.index[0]} → {s.index[-1]}  ({len(s)} months)")


if __name__ == "__main__":
    main()
