import pandas as pd
import pytest

import fetch_history as FH

SAMPLE = """This file was created using the CRSP database.
  Average Value Weighted Returns -- Monthly
,SMALL LoBM,SMALL HiBM
192607,   1.0866,  -0.1275
192608,   0.7831, -99.99
192609,  -2.8045,  -0.4399

  Average Equal Weighted Returns -- Monthly
,SMALL LoBM,SMALL HiBM
192607,   9.9,   9.9
"""


def test_first_monthly_table_parses_only_the_first_table():
    df = FH.first_monthly_table(SAMPLE)
    assert list(df.columns) == ["SMALL LoBM", "SMALL HiBM"]
    assert len(df) == 3  # stops at the blank line; equal-weighted table ignored
    assert df.index[0] == pd.Period("1926-07", "M")


def test_first_monthly_table_converts_percent_and_missing():
    df = FH.first_monthly_table(SAMPLE)
    assert df.loc["1926-07", "SMALL LoBM"] == pytest.approx(0.010866)
    assert pd.isna(df.loc["1926-08", "SMALL HiBM"])  # -99.99 marks missing
