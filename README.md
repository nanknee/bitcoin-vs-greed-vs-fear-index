# Bitcoin vs Fear & Greed API

FastAPI backend for the Bitcoin vs Fear & Greed calculator. It pulls the Fear & Greed
Index (alternative.me) and BTC prices (CoinGecko), merges them by day, and serves a signal
plus a historical analysis.

## Run

    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    uvicorn main:app --reload

Open http://127.0.0.1:8000/docs for interactive docs.

## Endpoints

| Endpoint | What it returns |
|---|---|
| `GET /api/signal?base=100&cost=60000&holdings=0.05` | Live index, zone, buy multiplier, suggested buy, P/L |
| `GET /api/history?days=365` | Daily price + index, merged by date (JSON) |
| `GET /api/analysis?days=365&horizon=30` | Correlations and forward return by zone |
| `GET /api/export.csv?days=365` | `date,price,index` rows to paste into the web calculator |

## Notes

- CoinGecko's free public API limits daily history to 365 days and rate-limits requests.
  Responses are cached for 5 minutes. If you add a demo API key, pass it as a header in `cached_get`.
- Zone averages use overlapping windows, so treat them as descriptive, not predictive.
- Not financial advice.

## Next steps for a portfolio version

1. Store the merged history in PostgreSQL and refresh it daily with a scheduler (APScheduler or cron).
2. Call `/api/signal` and `/api/analysis` from the web page instead of typing values in.
3. Add a backtest endpoint: compare a fixed-amount DCA with the sentiment-adjusted DCA.