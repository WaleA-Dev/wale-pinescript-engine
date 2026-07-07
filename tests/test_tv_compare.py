"""TV trade-list parser + comparison endpoint + export save."""
import web_app

TV_CSV = "﻿Trade #,Type,Date and time,Signal,Price USD,Position size (qty)\n" \
    "1,Exit long,2024-03-01 05:00,Close,110.5,1\n" \
    "1,Entry long,2024-01-10 05:00,Long,100.0,1\n" \
    "2,Exit long,2024-06-01 04:00,Close,95.0,1\n" \
    "2,Entry long,2024-05-01 04:00,Long,105.25,1\n"


def test_parse_tv_trades():
    tv = web_app._parse_tv_trades(TV_CSV)
    assert len(tv) == 2
    assert tv[0]["entry_date"].startswith("2024-01-10")
    assert tv[0]["exit_price"] == 110.5


def test_compare_trades_alignment_and_verdicts():
    tv = web_app._parse_tv_trades(TV_CSV)
    mine = [
        {"entry_date": "2024-01-10 05:00", "entry_price": 100.02,
         "exit_date": "2024-03-01 05:00", "exit_price": 110.4},
        {"entry_date": "2024-05-02 04:00", "entry_price": 105.5,   # one day off
         "exit_date": "2024-06-01 04:00", "exit_price": 95.1},
        {"entry_date": "2024-09-01 04:00", "entry_price": 90.0,    # extra
         "exit_date": None, "exit_price": None},
    ]
    r = web_app._compare_trades(tv, mine)
    assert r["legs"] == 4 and r["legs_exact"] == 3
    assert any(row.get("missing") == "tv" for row in r["rows"])
    exact = [row for row in r["rows"] if row.get("entry_match") and row.get("exit_match")]
    assert len(exact) == 1 and abs(exact[0]["entry_px_diff_pct"] - 0.02) < 0.001


def test_compare_endpoint_requires_backtest_and_parses_upload(tmp_path):
    c = web_app.app.test_client()
    web_app._state["last_trades"] = None
    r = c.post("/api/compare-tv")
    assert r.status_code == 400 and "backtest" in r.get_json()["error"]

    web_app._state["last_trades"] = [
        {"entry_date": "2024-01-10", "entry_price": 100.0,
         "exit_date": "2024-03-01", "exit_price": 110.5}]
    import io as _io
    r = c.post("/api/compare-tv",
               data={"file": (_io.BytesIO(TV_CSV.encode("utf-8-sig")), "tv.csv")},
               content_type="multipart/form-data")
    d = r.get_json()
    assert d["success"] and d["legs_exact"] >= 2


def test_export_save_writes_to_downloads(monkeypatch, tmp_path):
    monkeypatch.setattr(web_app.Path, "home", staticmethod(lambda: tmp_path))
    c = web_app.app.test_client()
    r = c.post("/api/export/save", json={"filename": "trades test.csv", "content": "a,b\n1,2\n"})
    d = r.get_json()
    assert d["success"]
    assert (tmp_path / "Downloads" / "trades_test.csv").exists()
