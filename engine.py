"""
Market making backtest on Polymarket's BTC 5-minute Up/Down markets.

Everything is expressed in "Up space": a trade on the Down token at price q is the
same economic trade as the opposite trade on the Up token at 1 - q.
  dir = +1  the taker bought Up (lifted the ask)  -> a maker sold Up
  dir = -1  the taker sold Up (hit the bid)        -> a maker bought Up
"""
import glob
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 300
FEE_RATE = 0.07      # taker fee = FEE_RATE * p * (1 - p) per share
REBATE_SHARE = 0.20  # share of taker fees paid back to makers


# ---------------------------------------------------------------- loading
def load_markets(path="data/markets/*.json"):
    """Returns (markets, trades). trades: one row per taker trade, in Up space.

    Reads data/markets.parquet + data/trades.parquet (the release files) when present,
    otherwise the raw per-market JSON downloads."""
    if Path("data/trades.parquet").exists():
        return _load_parquet()
    mk, tr = [], []
    for f in sorted(glob.glob(path)):
        d = json.load(open(f))
        if d["outcome_prices"] not in (["1", "0"], ["0", "1"]):
            continue
        y = 1 if d["outcome_prices"][0] == "1" else 0
        mk.append((d["start"], y, len(d["trades"])))
        for t in d["trades"]:
            up = t["outcome"] == "Up"
            buy = t["side"] == "BUY"
            tr.append((d["start"], t["timestamp"] - d["start"],
                       1 if up == buy else -1,
                       t["price"] if up else 1 - t["price"],
                       t["size"], t["price"] * t["size"], y))
    markets = pd.DataFrame(mk, columns=["start", "y", "n_trades"]).set_index("start")
    trades = pd.DataFrame(tr, columns=["start", "t", "dir", "x", "size", "notional", "y"])
    trades["x"] = trades.x.round(6)
    return markets, trades


def _load_parquet():
    m = pd.read_parquet("data/markets.parquet")
    m = m[m.outcome_up.isin(["0", "1"]) & (m.outcome_up != m.outcome_down)].copy()
    m["y"] = (m.outcome_up == "1").astype(int)
    t = pd.read_parquet("data/trades.parquet")
    t = t[t.start.isin(m.start)]
    m["n_trades"] = m.start.map(t.groupby("start").size()).fillna(0).astype(int)
    up = (t.outcome == "Up").to_numpy()
    buy = (t.side == "BUY").to_numpy()
    price = t.price.to_numpy()
    trades = pd.DataFrame({
        "start": t.start.to_numpy(), "t": (t.timestamp - t.start).to_numpy(),
        "dir": np.where(up == buy, 1, -1), "x": np.where(up, price, 1 - price),
        "size": t["size"].to_numpy(), "notional": price * t["size"].to_numpy(),
        "y": t.start.map(m.set_index("start").y).to_numpy(),
    })
    trades["x"] = trades.x.round(6)
    return m.set_index("start")[["y", "n_trades"]], trades


def load_spot(path="data/spot/*.parquet"):
    s = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(path))).drop_duplicates("ts").set_index("ts").close
    s = s.reindex(range(s.index.min(), s.index.max() + 1)).ffill()
    lr = np.log(s).diff()
    vol = lr.rolling(1800, min_periods=600).std().shift(1)   # per-second sigma, past 30 min only
    return s, vol


# ---------------------------------------------------------------- resolution rules
RULE = "twap60_end_vs_twap60_start"  # matches 97.5% of actual resolutions with Binance data
RULES = {
    # name: (target interval [a, b) relative to start, reference interval or point)
    "twap_window_vs_open": ((0, WINDOW), (0, 1)),
    "close_vs_open": ((WINDOW - 1, WINDOW), (0, 1)),
    "twap60_end_vs_twap60_start": ((WINDOW - 60, WINDOW), (-60, 0)),
    "twap_window_vs_twap60_start": ((0, WINDOW), (-60, 0)),
}


def rule_outcome(spot, start, rule):
    (a, b), (ra, rb) = RULES[rule]
    tgt = spot.loc[start + a: start + b - 1].mean()
    ref = spot.loc[start + ra: start + rb - 1].mean()
    return int(tgt >= ref)


# ---------------------------------------------------------------- fair value
def _ncdf(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def fair_path(spot, vol, start, rule):
    """P(Up) at every second s = 0..WINDOW given spot up to s (Brownian approximation).

    The target is the average of S over [a, b). At time s the part already observed is
    known; the future part is Gaussian with mean S_s and variance from integrated BM.
    """
    (a, b), (ra, rb) = RULES[rule]
    S = spot.loc[start - 60: start + WINDOW].to_numpy()
    off = 60
    sig = vol.get(start, np.nan)
    if np.isnan(sig) or len(S) < off + WINDOW + 1:
        return None
    ref = S[off + ra: off + rb].mean()
    out = np.empty(WINDOW + 1)
    n = b - a
    for s in range(WINDOW + 1):
        st = S[off + s]
        known_hi = min(max(s, a), b)
        known = S[off + a: off + known_hi].sum() if known_hi > a else 0.0
        d1, d2 = max(a - s, 0), b - s          # future part [max(a,s), b) measured from s
        D = d2 - d1 if d2 > d1 else 0
        mean = (known + st * D) / n
        var = (st * sig) ** 2 * (D * D * d1 + D ** 3 / 3) / n ** 2 if D > 0 else 0.0
        if var <= 0:
            out[s] = 1.0 if mean >= ref else 0.0
        else:
            out[s] = _ncdf((mean - ref) / sqrt(var))
    return out


# ---------------------------------------------------------------- quoting simulation
def naive_path(tr_window):
    """Center for the naive quoter: EWMA of Up-space trade prices, sampled at each second."""
    out = np.full(WINDOW + 1, np.nan)
    ew, j = np.nan, 0
    t, x = tr_window.t.to_numpy(), tr_window.x.to_numpy()
    for s in range(WINDOW + 1):
        while j < len(t) and t[j] <= s:
            ew = x[j] if np.isnan(ew) else 0.7 * ew + 0.3 * x[j]
            j += 1
        out[s] = ew
    return out


def simulate_market(tr, fair, y, center="spot", half=0.02, latency=1, through=0.01,
                    size=20.0, max_inv=100.0, gamma=1.0, stop_before=15, band=(0.04, 0.96),
                    pause=None, pause_mode="all"):
    """One market. tr: taker trades inside the window (t, dir, x, size), sorted by t.

    Quotes at second s use information from second s - latency. A resting bid fills when a
    taker sale prints at or below bid - through (it swept our level), and fills our full
    remaining size up to the traded size. Inventory is held to resolution.
    pause: optional boolean array per second; quoting stops at s when pause[s - latency] is set.
    pause_mode: "all" stops both sides; "entry" keeps only the side that reduces inventory.
    Returns (pnl, rebate, fills list).
    """
    ref = fair if center == "spot" else naive_path(tr)
    t, d, x, q = (tr[c].to_numpy() for c in ("t", "dir", "x", "size"))
    cash = inv = rebate = 0.0
    fills = []
    j = 0
    n = len(t)
    while j < n and t[j] < 0:
        j += 1
    for s in range(0, WINDOW):
        k = s - latency
        c = ref[k] if k >= 0 else np.nan
        quoting = not np.isnan(c) and band[0] <= c <= band[1] and s < WINDOW - stop_before
        paused = quoting and pause is not None and k >= 0 and pause[k]
        if paused and pause_mode == "all":
            quoting = False
        if quoting:
            skew = gamma * half * inv / max_inv
            bid = np.floor((c - half - skew) * 100 + 1e-9) / 100
            ask = np.ceil((c + half - skew) * 100 - 1e-9) / 100
            bid_on, ask_on = bid >= 0.01 and inv < max_inv, ask <= 0.99 and inv > -max_inv
            if paused:                                  # entry mode: only quote the side that reduces inventory
                bid_on, ask_on = bid_on and inv < 0, ask_on and inv > 0
        else:
            bid_on = ask_on = False
        while j < n and t[j] == s:
            if bid_on and d[j] == -1 and x[j] <= bid - through + 1e-9:
                f = min(size, q[j], max_inv - inv)
                cash -= bid * f; inv += f
                rebate += REBATE_SHARE * FEE_RATE * bid * (1 - bid) * f
                fills.append((s, 1, bid, f))
                bid_on = False
            elif ask_on and d[j] == 1 and x[j] >= ask + through - 1e-9:
                f = min(size, q[j], max_inv + inv)
                cash += ask * f; inv -= f
                rebate += REBATE_SHARE * FEE_RATE * ask * (1 - ask) * f
                fills.append((s, -1, ask, f))
                ask_on = False
            j += 1
    pnl = cash + inv * y
    return pnl, rebate, fills
