"""
Alpaca Market Data provider — free live/historical OHLCV bars.

Uses the free IEX feed on data.alpaca.markets (any free Alpaca account works).
Two auth styles are supported with the same key/secret pair:

1. Classic header auth:   APCA-API-KEY-ID / APCA-API-SECRET-KEY
2. OAuth2 client_credentials via https://authx.alpaca.markets/v1/oauth2/token
   (the key ID is the client_id, the secret is the client_secret; the returned
   Bearer token is used against the data API and refreshed automatically).

Keys are stored locally in ~/.wale_backtest/config.json — never bundled in
the EXE. Users without keys get WALE_TRIAL_LIMIT free data downloads
(served from Yahoo Finance), after which a free Alpaca key is required.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

DATA_BASE = "https://data.alpaca.markets/v2"
AUTHX_TOKEN_URL = "https://authx.alpaca.markets/v1/oauth2/token"

CONFIG_DIR = Path.home() / ".wale_backtest"
CONFIG_FILE = CONFIG_DIR / "config.json"

TRIAL_LIMIT = 5

# UI interval -> Alpaca timeframe
TIMEFRAME_MAP = {
    "1min": "1Min", "5min": "5Min", "15min": "15Min", "30min": "30Min",
    "1h": "1Hour", "4h": "4Hour", "1d": "1Day", "1wk": "1Week",
    "1Min": "1Min", "5Min": "5Min", "15Min": "15Min", "30Min": "30Min",
    "1Hour": "1Hour", "4Hour": "4Hour", "1Day": "1Day", "1Week": "1Week",
}

# In-memory OAuth token cache: {"token": str, "expires_at": epoch}
_token_cache: dict = {}


# ── Config storage ───────────────────────────────────────────────────────────

def load_config() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def save_keys(key_id: str, secret: str) -> None:
    cfg = load_config()
    cfg["alpaca_key_id"] = key_id.strip()
    cfg["alpaca_secret"] = secret.strip()
    cfg.pop("alpaca_auth_mode", None)  # re-detect on next request
    save_config(cfg)
    _token_cache.clear()


def get_keys() -> tuple[Optional[str], Optional[str]]:
    cfg = load_config()
    return cfg.get("alpaca_key_id") or None, cfg.get("alpaca_secret") or None


def clear_keys() -> None:
    cfg = load_config()
    for k in ("alpaca_key_id", "alpaca_secret", "alpaca_auth_mode"):
        cfg.pop(k, None)
    save_config(cfg)
    _token_cache.clear()


def has_keys() -> bool:
    key_id, secret = get_keys()
    return bool(key_id and secret)


# ── Free trial (no-key) tracking ─────────────────────────────────────────────

def trial_remaining() -> int:
    used = int(load_config().get("trial_fetches_used", 0))
    return max(0, TRIAL_LIMIT - used)


def consume_trial() -> int:
    """Consume one trial fetch, return remaining count."""
    cfg = load_config()
    cfg["trial_fetches_used"] = int(cfg.get("trial_fetches_used", 0)) + 1
    save_config(cfg)
    return max(0, TRIAL_LIMIT - cfg["trial_fetches_used"])


# ── Auth ─────────────────────────────────────────────────────────────────────

class AlpacaAuthError(Exception):
    pass


def _fetch_oauth_token(key_id: str, secret: str) -> str:
    """Exchange client credentials for a Bearer token via authx."""
    now = time.time()
    if _token_cache.get("token") and now < _token_cache.get("expires_at", 0) - 30:
        return _token_cache["token"]

    resp = requests.post(
        AUTHX_TOKEN_URL,
        headers={"accept": "application/json",
                 "content-type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials",
              "client_id": key_id, "client_secret": secret},
        timeout=15,
    )
    if resp.status_code != 200:
        try:
            err = resp.json().get("error", resp.text)
        except ValueError:
            err = resp.text
        raise AlpacaAuthError(f"OAuth token exchange failed ({resp.status_code}): {err}")

    body = resp.json()
    token = body["access_token"]
    _token_cache["token"] = token
    _token_cache["expires_at"] = now + int(body.get("expires_in", 900))
    return token


def _auth_headers(key_id: str, secret: str, mode: str) -> dict:
    if mode == "oauth":
        return {"Authorization": f"Bearer {_fetch_oauth_token(key_id, secret)}"}
    return {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}


def _detect_auth_mode(key_id: str, secret: str) -> str:
    """Figure out which auth style this key pair uses; cache the answer."""
    cfg = load_config()
    cached = cfg.get("alpaca_auth_mode")
    if cached in ("headers", "oauth"):
        return cached

    probe = f"{DATA_BASE}/stocks/bars/latest"
    for mode in ("headers", "oauth"):
        try:
            r = requests.get(
                probe, params={"symbols": "AAPL", "feed": "iex"},
                headers=_auth_headers(key_id, secret, mode), timeout=15,
            )
            if r.status_code == 200:
                cfg["alpaca_auth_mode"] = mode
                save_config(cfg)
                return mode
        except AlpacaAuthError:
            continue
        except requests.RequestException:
            raise
    raise AlpacaAuthError(
        "Alpaca rejected these credentials with both header auth and OAuth. "
        "Check your API Key ID / Secret (get free keys at alpaca.markets)."
    )


def validate_keys(key_id: str, secret: str) -> dict:
    """Test credentials against the data API. Returns {valid, mode|error}."""
    try:
        cfg = load_config()
        cfg.pop("alpaca_auth_mode", None)
        save_config(cfg)
        _token_cache.clear()

        # Temporarily test without persisting keys
        probe = f"{DATA_BASE}/stocks/bars/latest"
        for mode in ("headers", "oauth"):
            try:
                r = requests.get(
                    probe, params={"symbols": "AAPL", "feed": "iex"},
                    headers=_auth_headers(key_id, secret, mode), timeout=15,
                )
                if r.status_code == 200:
                    return {"valid": True, "mode": mode}
            except AlpacaAuthError:
                continue
        return {"valid": False,
                "error": "Credentials rejected by Alpaca (tried header auth and OAuth)."}
    except requests.RequestException as e:
        return {"valid": False, "error": f"Network error: {e}"}


# ── Bars ─────────────────────────────────────────────────────────────────────

def fetch_bars(
    symbol: str,
    timeframe: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    feed: str = "iex",
    adjustment: str = "split",
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from Alpaca (free IEX feed), paginated.
    Returns DataFrame indexed by UTC timestamp with open/high/low/close/volume.

    adjustment: "split" matches TradingView's default chart; "all" (splits +
    dividends) matches a TV chart with the "adj" dividend toggle enabled.
    """
    if adjustment not in ("raw", "split", "dividend", "all"):
        raise ValueError(f"Unsupported adjustment: {adjustment}")
    key_id, secret = get_keys()
    if not (key_id and secret):
        raise AlpacaAuthError("No Alpaca API keys saved. Add your free Alpaca key first.")

    tf = TIMEFRAME_MAP.get(timeframe)
    if tf is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    symbol = symbol.strip().upper()
    now_utc = datetime.now(timezone.utc)

    if not start:
        # sensible defaults: intraday ~2y, daily ~10y
        days = 730 if tf not in ("1Day", "1Week") else 3650
        start_dt = now_utc - timedelta(days=days)
    else:
        start_dt = pd.Timestamp(start, tz="UTC").to_pydatetime()

    # Free plan can't query the most recent 15 minutes of SIP; IEX is fine but
    # clamp anyway to avoid 403 subscription errors near real-time.
    end_dt = pd.Timestamp(end, tz="UTC").to_pydatetime() if end else now_utc
    end_dt = min(end_dt, now_utc - timedelta(minutes=16))

    mode = _detect_auth_mode(key_id, secret)
    url = f"{DATA_BASE}/stocks/{symbol}/bars"
    params = {
        "timeframe": tf,
        "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 10000,
        "adjustment": adjustment,
        "feed": feed,
    }

    rows = []
    page_token = None
    while True:
        if page_token:
            params["page_token"] = page_token
        r = requests.get(url, params=params,
                         headers=_auth_headers(key_id, secret, mode), timeout=30)
        if r.status_code in (401, 403):
            raise AlpacaAuthError(
                f"Alpaca rejected the request ({r.status_code}): {r.text[:200]}. "
                "Free accounts must use the IEX feed."
            )
        if r.status_code == 422:
            raise ValueError(f"Alpaca rejected parameters: {r.text[:300]}")
        if r.status_code == 429:
            time.sleep(3)
            continue
        r.raise_for_status()
        body = r.json()
        rows.extend(body.get("bars") or [])
        page_token = body.get("next_page_token")
        if not page_token:
            break

    if not rows:
        raise ValueError(
            f"No data returned for {symbol} ({tf}) in that range. "
            "Check the ticker; note IEX history is thinner before 2017."
        )

    df = pd.DataFrame(rows)
    df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high",
                            "l": "low", "c": "close", "v": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df
