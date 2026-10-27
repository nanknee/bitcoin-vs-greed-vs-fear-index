"""Bitcoin vs Fear & Greed API (Flask version, ready to deploy).

Local:      python app.py
Production: gunicorn app:app
"""
import os
import time
from typing import Any, Optional

import httpx
import pandas as pd
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

FNG_URL = "https://api.alternative.me/fng/"
CG_URL = "https://api.coingecko.com/api/v3"
CACHE_TTL = 300  # seconds; keeps you inside free-tier rate limits

# (name, upper bound of index, buy multiplier) -- same rules as the web calculator
ZONES = [
    ("Extreme fear", 24, 1.5),
    ("Fear", 44, 1.25),
    ("Neutral", 55, 1.0),
    ("Greed", 75, 0.75),
    ("Extreme greed", 100, 0.5),
]

app = Flask(__name__)
CORS(app)  # allow your web page to call this API from another domain

_cache: dict[str, tuple[float, Any]] = {}


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


@app.errorhandler(ApiError)
def handle_api_error(err: ApiError):
    return jsonify({"error": err.message}), err.status


def arg_float(name: str, default: Optional[float], lo: float = 0, required_positive: bool = False):
    raw = request.args.get(name)
    if raw in (None, ""):
        return default
    try:
        val = float(raw)
    except ValueError:
        raise ApiError(f"'{name}' must be a number.")
    if val < lo or (required_positive and val <= 0):
        raise ApiError(f"'{name}' is out of range.")
    return val


def arg_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = request.args.get(name)
    if raw in (None, ""):
        return default
    try:
        val = int(raw)
    except ValueError:
        raise ApiError(f"'{name}' must be a whole number.")
    if not lo <= val <= hi:
        raise ApiError(f"'{name}' must be between {lo} and {hi}.")
    return val


def cached_get(url: str, params: dict) -> Any:
    """GET with a small in-memory TTL cache and clean error handling."""
    key = url + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    try:
        resp = httpx.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ApiError(f"Upstream request failed: {exc}", 502)
    _cache[key] = (time.time(), data)
    return data


def zone_of(value: float) -> tuple[str, float]:
    for name, upper, mult in ZONES:
        if value <= upper:
            return name, mult
    return ZONES[-1][0], ZONES[-1][2]


def fetch_fng(days: int) -> pd.DataFrame:
    rows = cached_get(FNG_URL, {"limit": days, "format": "json"}).get("data", [])
    df = pd.DataFrame(
        {
            "date": [pd.to_datetime(int(r["timestamp"]), unit="s", utc=True).date() for r in rows],
            "fng": [float(r["value"]) for r in rows],
        }
    )
    return df.drop_duplicates("date").sort_values("date")


def fetch_prices(days: int) -> pd.DataFrame:
    data = cached_get(
        f"{CG_URL}/coins/bitcoin/market_chart",
        {"vs_currency": "usd", "days": days, "interval": "daily"},
    )
    df = pd.DataFrame(data.get("prices", []), columns=["ts", "price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.date
    # keep the last reading per day (CoinGecko adds a partial point for today)
    return df.drop_duplicates("date", keep="last")[["date", "price"]]


def merged_history(days: int) -> pd.DataFrame:
    df = fetch_prices(days).merge(fetch_fng(days + 5), on="date", how="inner")
    if df.empty:
        raise ApiError("No overlapping Fear & Greed and price data returned.", 502)
    return df.sort_values("date").reset_index(drop=True)


def num(x) -> Optional[float]:
    """JSON-safe number: NaN becomes null."""
    return None if pd.isna(x) else round(float(x), 4)


def analyze(df: pd.DataFrame, horizon: int) -> dict:
    df = df.copy()
    df["ret_next"] = df["price"].pct_change().shift(-1)
    df["fwd"] = df["price"].shift(-horizon) / df["price"] - 1
    df["zone"] = df["fng"].map(lambda v: zone_of(v)[0])

    zones = []
    for name, _, _ in ZONES:
        fwd = df.loc[df["zone"] == name, "fwd"].dropna()
        zones.append(
            {
                "zone": name,
                "days": int(len(fwd)),
                "avg_forward_return": num(fwd.mean()) if len(fwd) else None,
                "win_rate": num((fwd > 0).mean()) if len(fwd) else None,
            }
        )
    return {
        "days_analyzed": int(len(df)),
        "horizon_days": horizon,
        "corr_index_vs_price": num(df["fng"].corr(df["price"])),
        "corr_index_vs_next_day_return": num(df["fng"].corr(df["ret_next"])),
        "zones": zones,
    }


@app.get("/")
def index():
    return jsonify(
        {
            "name": "Bitcoin vs Fear & Greed API",
            "endpoints": ["/api/signal", "/api/history", "/api/analysis", "/api/export.csv", "/health"],
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/signal")
def signal():
    """Live index + price, the zone, and a sentiment-adjusted buy amount."""
    base = arg_float("base", 100)
    cost = arg_float("cost", None, required_positive=True)
    holdings = arg_float("holdings", None)

    latest = cached_get(FNG_URL, {"limit": 1, "format": "json"})["data"][0]
    value = float(latest["value"])
    price = cached_get(f"{CG_URL}/simple/price", {"ids": "bitcoin", "vs_currencies": "usd"})["bitcoin"]["usd"]

    name, mult = zone_of(value)
    buy = base * mult
    out = {
        "fng": value,
        "zone": name,
        "btc_price_usd": price,
        "multiplier": mult,
        "suggested_buy_usd": round(buy, 2),
        "btc_for_suggested_buy": round(buy / price, 8),
    }
    if cost:
        out["price_vs_cost_pct"] = round((price / cost - 1) * 100, 2)
        if holdings:
            out["unrealized_pnl_usd"] = round((price - cost) * holdings, 2)
    return jsonify(out)


@app.get("/api/history")
def history():
    """Daily BTC price and Fear & Greed value, merged by date."""
    df = merged_history(arg_int("days", 365, 40, 365))
    return jsonify([{"date": str(r.date), "price": round(r.price, 2), "fng": r.fng} for r in df.itertuples()])


@app.get("/api/analysis")
def analysis():
    """Correlations and average forward return by sentiment zone."""
    days = arg_int("days", 365, 40, 365)
    horizon = arg_int("horizon", 30, 1, 90)
    return jsonify(analyze(merged_history(days), horizon))


@app.get("/api/export.csv")
def export_csv():
    """date,price,index rows -- paste straight into the calculator's Historical check."""
    df = merged_history(arg_int("days", 365, 40, 365))
    body = "\n".join(f"{r.date},{r.price:.2f},{int(r.fng)}" for r in df.itertuples())
    return Response(body, mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)