import pytest

import indicators as I

HTML = """<table><tr><td>Oct 22, 2026</td>
<td>&#x2002;40.58</td></tr><tr><td>Oct 1, 2026</td><td>
 40.81 </td></tr></table>"""


def test_parse_cape_table():
    assert I.parse_cape_table(HTML) == [("Oct 22, 2026", 40.58), ("Oct 1, 2026", 40.81)]
    assert I.parse_cape_table("<html>no table</html>") == []


@pytest.mark.parametrize("value,expected", [(40.6, "concentration persists"), (27.0, "SUPPORTS SHRINKING"),
                                            (28.0, "SUPPORTS SHRINKING")])
def test_status_threshold(value, expected):
    assert I.status(value, {"baseline": 40.6, "shrink_below": 28.0}) == expected


def test_top10_and_core_ai_on_fake_holdings(fake_root):
    # fake IVV: NVDA 60%, AAPL 40% -> top-10 = 100%; fake ITOT: NVDA (core) 50%
    assert I.top10_share("IVV") == pytest.approx(1.0)
    assert I.core_ai_share("ITOT") == pytest.approx(0.5)
