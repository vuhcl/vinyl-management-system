"""Tests for Discogs sale history HTML parsing."""

from __future__ import annotations

from price_estimator.src.scrape.discogs_sale_history_parse import (
    looks_like_login_or_challenge,
    parse_sale_history_html,
    sale_history_url,
)
from price_estimator.src.storage.sale_history_db import SaleHistoryDB


FIXTURE_HTML = """
<html><body><div id="page_content">
<ul>
<li>Last sold on: Apr 10, 2026</li>
<li>Average: $140.06</li>
<li>Median: $117.65</li>
<li>High: $294.12</li>
<li>Low: $72.46</li>
</ul>
<table><thead><tr>
<th>Order Date</th><th>Condition</th><th>Sleeve Condition</th>
<th>Price in your currency</th><th>Price</th>
</tr></thead><tbody>
<tr>
  <td>2026-04-10</td><td>Mint (M)</td><td>Near Mint (NM or M-)</td>
  <td>$150.00</td><td>CA$225.00</td>
</tr>
<tr><td colspan="5">In hand ready to ship!</td></tr>
<tr>
  <td>2026-04-09</td><td>Near Mint (NM or M-)</td><td>Mint (M)</td>
  <td>$163.04</td><td>€135.00</td>
</tr>
<tr><td colspan="5">Sealed</td></tr>
</tbody></table>
</div></body></html>
"""


def test_looks_like_login_or_challenge() -> None:
    assert looks_like_login_or_challenge("<html>Sign In</html><input name=password>") is True
    assert looks_like_login_or_challenge(FIXTURE_HTML) is False


def test_sale_history_url() -> None:
    assert sale_history_url("36946488") == (
        "https://www.discogs.com/sell/history/36946488"
    )


def test_parse_sale_history_fixture() -> None:
    p = parse_sale_history_html(FIXTURE_HTML, "36946488")
    assert p.release_id == "36946488"
    assert p.summary is not None
    assert p.summary.median == 117.65
    assert p.summary.low == 72.46
    assert len(p.rows) == 2
    assert p.rows[0].order_date == "2026-04-10"
    assert p.rows[0].media_condition == "Mint (M)"
    assert p.rows[0].seller_comments == "In hand ready to ship!"
    assert "CA$225.00" in p.rows[0].price_original_text
    assert p.rows[1].seller_comments == "Sealed"
    assert p.rows[0].row_hash("36946488") != p.rows[1].row_hash("36946488")


def test_sale_history_db_upsert(tmp_path) -> None:
    db = SaleHistoryDB(tmp_path / "h.sqlite")
    p = parse_sale_history_html(FIXTURE_HTML, "1")
    db.upsert_parsed(p, status="ok")
    st = db.last_status("1")
    assert st is not None
    assert st["status"] == "ok"
    assert st["num_rows"] == 2
    assert db.should_skip_resume("1", ok_hours=24.0) is True
