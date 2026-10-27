# %% [markdown]
# # Bitcoin vs Fear & Greed: data science analysis
# Questions: (1) does the index move with price, (2) does it lead or lag price,
# (3) what has BTC returned after each sentiment zone, (4) does sentiment-weighted
# DCA beat fixed DCA, (5) can the index help predict 30-day direction?
#
# Run as a script, or open in VS Code / Jupyter (cells are split by `# %%`).

# %%
import os
import httpx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = 30          # forward-return window in days
BASE_BUY = 100        # normal daily buy in USD for the DCA backtest
DEMO = os.environ.get("BTC_FNG_DEMO") == "1"   # synthetic data, for offline testing

ZONES = [  # (name, upper bound, buy multiplier)
    ("Extreme fear", 24, 1.5),
    ("Fear", 44, 1.25),
    ("Neutral", 55, 1.0),
    ("Greed", 75, 0.75),
    ("Extreme greed", 100, 0.5),
]
ZONE_NAMES = [z[0] for z in ZONES]


def zone_of(v):
    for name, upper, mult in ZONES:
        if v <= upper:
            return name, mult
    return ZONES[-1][0], ZONES[-1][2]


# %% [markdown]
# ## 1. Load and merge the data
# Fear & Greed: alternative.me (full history, daily). Price: Yahoo Finance via `yfinance`
# (`pip install yfinance`). To use your own file instead, build `price` from a CSV.

# %%
def load_fng():
    rows = httpx.get(
        "https://api.alternative.me/fng/", params={"limit": 0, "format": "json"}, timeout=30
    ).json()["data"]
    s = pd.Series(
        {pd.to_datetime(int(r["timestamp"]), unit="s").normalize(): float(r["value"]) for r in rows},
        name="fng",
    )
    return s.sort_index()


def load_price():
    import yfinance as yf
    px = yf.download("BTC-USD", start="2018-02-01", progress=False, auto_adjust=True)["Close"].squeeze()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    return px.rename("price")


def synthetic():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2019-01-01", periods=1500)
    price = pd.Series(10000 * np.exp(np.cumsum(rng.normal(0.0008, 0.03, len(idx)))), idx, name="price")
    fng = (50 + 350 * price.pct_change(14).rolling(5).mean().fillna(0) + rng.normal(0, 8, len(idx))).clip(0, 100)
    return price, fng.rename("fng")


price, fng = synthetic() if DEMO else (load_price(), load_fng())

df = pd.concat([price, fng], axis=1, join="inner").dropna()
df["ret"] = df["price"].pct_change()
df["fwd"] = df["price"].shift(-HORIZON) / df["price"] - 1
df["zone"] = pd.Categorical(df["fng"].map(lambda v: zone_of(v)[0]), categories=ZONE_NAMES, ordered=True)

print(f"{len(df)} days, {df.index.min().date()} to {df.index.max().date()}")
print(df[["price", "fng", "ret", "fwd"]].describe().round(3))

# %% [markdown]
# ## 2. Price and sentiment over time

# %%
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
a1.plot(df.index, df["price"], lw=1)
a1.set_yscale("log")
a1.set_ylabel("BTC price (USD, log)")
a2.plot(df.index, df["fng"], lw=0.8, color="tab:orange")
for edge in (25, 75):
    a2.axhline(edge, color="grey", ls=":", lw=0.8)
a2.set_ylabel("Fear & Greed")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Correlation and lead-lag
# Level correlation is inflated because both series trend. The lead-lag chart is more honest:
# k < 0 means sentiment today vs the return k days *before* (sentiment follows price);
# k > 0 means sentiment today vs the return k days *after* (sentiment predicts price).

# %%
print("Pearson  (index vs price level):", round(df["fng"].corr(df["price"]), 3))
print("Spearman (index vs price level):", round(df["fng"].corr(df["price"], method="spearman"), 3))

lags = range(-10, 11)
cc = pd.Series({k: df["fng"].corr(df["ret"].shift(-k)) for k in lags})
cc.plot.bar(figsize=(10, 3.5), title="corr(index today, daily return k days later)")
plt.xlabel("k (days)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. What did BTC do after each zone?

# %%
def zone_table(h):
    fwd = df["price"].shift(-h) / df["price"] - 1
    g = fwd.groupby(df["zone"], observed=False)
    return pd.DataFrame({
        "days": g.count(),
        "mean": g.mean(),
        "median": g.median(),
        "win_rate": g.apply(lambda s: (s.dropna() > 0).mean()),
    })


for h in (7, 30, 90):
    print(f"\nForward {h}-day return by zone")
    print(zone_table(h).round(3))

plt.figure(figsize=(9, 4))
plt.boxplot([df.loc[df["zone"] == z, "fwd"].dropna() for z in ZONE_NAMES], showfliers=False)
plt.xticks(range(1, len(ZONE_NAMES) + 1), ZONE_NAMES)
plt.axhline(0, color="grey", lw=0.8)
plt.title(f"{HORIZON}-day forward return by zone")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Is the difference between zones statistically real?
# Overlapping 30-day windows are not independent, so test on a non-overlapping sample.

# %%
s = df.iloc[::HORIZON].dropna(subset=["fwd"])
groups = [s.loc[s["zone"] == z, "fwd"] for z in ZONE_NAMES if (s["zone"] == z).sum() >= 2]
h_stat, p_kw = stats.kruskal(*groups)
rho, p_rho = stats.spearmanr(s["fng"], s["fwd"])
print(f"Non-overlapping samples: {len(s)}")
print(f"Kruskal-Wallis across zones: H={h_stat:.2f}, p={p_kw:.3f}")
print(f"Spearman index vs forward return: rho={rho:.3f}, p={p_rho:.3f}")

# %% [markdown]
# ## 5. Backtest: fixed DCA vs sentiment-weighted DCA
# Sentiment DCA spends more in fear and less in greed, so it invests a different total.
# Compare average cost per BTC, and also an equal-budget version scaled to the same total spend.

# %%
d = df.copy()
d["mult"] = d["fng"].map(lambda v: zone_of(v)[1])
last = d["price"].iloc[-1]


def evaluate(spend):
    coins = (spend / d["price"]).sum()
    invested = spend.sum()
    return {"invested": invested, "btc": coins, "avg_cost": invested / coins,
            "final_value": coins * last, "roi": coins * last / invested - 1}


fixed = pd.Series(BASE_BUY, index=d.index, dtype=float)
sent = BASE_BUY * d["mult"]
sent_eq = sent * fixed.sum() / sent.sum()   # same total budget as fixed DCA

res = pd.DataFrame({
    "Fixed DCA": evaluate(fixed),
    "Sentiment DCA": evaluate(sent),
    "Sentiment DCA (equal budget)": evaluate(sent_eq),
}).T
print(res.round(4))

plt.figure(figsize=(10, 4))
for label, spend in (("Fixed DCA", fixed), ("Sentiment DCA (equal budget)", sent_eq)):
    plt.plot(d.index, (spend / d["price"]).cumsum() * d["price"], label=label)
plt.legend()
plt.ylabel("Portfolio value (USD)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Can sentiment help predict 30-day direction?
# Logistic regression on index features, validated with a time-ordered split and a gap equal
# to the horizon so future data never leaks into training. Compared with always guessing the
# majority class from the training period.

# %%
feat = pd.DataFrame({
    "fng": df["fng"],
    "fng_chg7": df["fng"].diff(7),
    "fng_ma30": df["fng"].rolling(30).mean(),
    "mom30": df["price"].pct_change(30),
}, index=df.index)
data = feat.join(df["fwd"]).dropna()
X, y = data[feat.columns], (data["fwd"] > 0).astype(int)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
acc, base = [], []
for tr, te in TimeSeriesSplit(n_splits=5, gap=HORIZON).split(X):
    model.fit(X.iloc[tr], y.iloc[tr])
    acc.append(model.score(X.iloc[te], y.iloc[te]))
    base.append((y.iloc[te] == int(y.iloc[tr].mean() > 0.5)).mean())

print(f"Model accuracy by fold : {np.round(acc, 3)}  (mean {np.mean(acc):.3f})")
print(f"Majority-class baseline: {np.round(base, 3)}  (mean {np.mean(base):.3f})")
print("If the model does not clearly beat the baseline, the index adds little predictive signal here.")