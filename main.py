"""Bitcoin vs Fear & Greed API.

Pulls the Crypto Fear & Greed Index (alternative.me) and BTC prices (CoinGecko),
merges them by day, and serves a sentiment signal plus a historical analysis.

Run:  uvicorn main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import time
from typing import Any, Optional

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

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

app = FastAPI(title="Bitcoin vs Fear & Greed API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"]
)

_cache: dict[str, tuple[float, Any]] = {}


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
        raise HTTPException(502, f"Upstream request failed: {exc}")
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
            "date": [
                pd.to_datetime(int(r["timestamp"]), unit="s", utc=True).date()
                for r in rows
            ],
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
        raise HTTPException(502, "No overlapping Fear & Greed and price data returned.")
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


@app.get("/api/signal")
def signal(
    base: float = Query(100, ge=0, description="Your normal buy amount in USD"),
    cost: Optional[float] = Query(None, gt=0, description="Your average cost in USD"),
    holdings: Optional[float] = Query(None, ge=0, description="BTC you hold"),
):
    """Live index + price, the zone, and a sentiment-adjusted buy amount."""
    latest = cached_get(FNG_URL, {"limit": 1, "format": "json"})["data"][0]
    value = float(latest["value"])
    price = cached_get(
        f"{CG_URL}/simple/price", {"ids": "bitcoin", "vs_currencies": "usd"}
    )["bitcoin"]["usd"]

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
    return out


@app.get("/api/history")
def history(days: int = Query(365, ge=40, le=365)):
    """Daily BTC price and Fear & Greed value, merged by date."""
    df = merged_history(days)
    return [
        {"date": str(r.date), "price": round(r.price, 2), "fng": r.fng}
        for r in df.itertuples()
    ]


@app.get("/api/analysis")
def analysis(
    days: int = Query(365, ge=40, le=365),
    horizon: int = Query(30, ge=1, le=90, description="Forward-return window in days"),
):
    """Correlations and average forward return by sentiment zone."""
    return analyze(merged_history(days), horizon)


@app.get("/api/export.csv", response_class=PlainTextResponse)
def export_csv(days: int = Query(365, ge=40, le=365)):
    """date,price,index rows -- paste straight into the calculator's Historical check."""
    df = merged_history(days)
    return "\n".join(
        f"{r.date},{r.price:.2f},{int(r.fng)}" for r in df.itertuples()
    )