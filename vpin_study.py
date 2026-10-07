"""
Does VPIN predict toxic flow on Polymarket's BTC 5-minute markets?

For every in-window taker trade we compute VPIN from the trades *before* it, then
measure what the maker on the other side earned against the spot fair value.

VPIN variants
  bot      the tracker my bot used: Up-token trades only, $5 buckets, 20 buckets,
           sides as reported for the Up token (main.py passed vpin_up everywhere)
  flow_B   both tokens mapped to Up space (a Down buy is an Up sell), $B buckets, 20 buckets

Timing: Data API timestamps are ~2.2 s after the match, so "at the fill" uses the
fair value 2 s before the reported timestamp.
"""
import json
from collections import deque
from multiprocessing import Pool

import numpy as np
import pandas as pd

import engine as E

DELAY = 2
N_BUCKETS = 20
FLOW_SIZES = (5.0, 50.0, 500.0)


class VPIN:
    """Same bucketing as the bot's VPINTracker (overflow split at the current buy/sell mix)."""

    def __init__(self, bucket, n=N_BUCKETS):
        self.bucket, self.buy, self.sell, self.vol = bucket, 0.0, 0.0, 0.0
        self.b = deque(maxlen=n)

    def add(self, notional, is_buy):
        if is_buy:
            self.buy += notional
        else:
            self.sell += notional
        self.vol += notional
        while self.vol >= self.bucket:
            r = self.bucket / self.vol
            bb, ss = self.buy * r, self.sell * r
            self.b.append((bb, ss))
            self.buy -= bb; self.sell -= ss; self.vol -= self.bucket

    def value(self):
        if len(self.b) < 2:
            return np.nan
        return sum(abs(x - y) / max(x + y, 1e-9) for x, y in self.b) / len(self.b)

    def signed(self):
        if len(self.b) < 2:
            return np.nan
        return sum((x - y) / max(x + y, 1e-9) for x, y in self.b) / len(self.b)


def market_features(args):
    start, g, fair, y = args
    rows = []
    bot = VPIN(5.0)
    flows = {B: VPIN(B) for B in FLOW_SIZES}
    sig50 = flows[50.0]
    recent, recent_sum = deque(), 0.0                     # (t, notional) over the last 10 s
    marks = []                                           # (t, bot VPIN after the trade)
    for t, outcome, side, price, size in g.itertuples(index=False):
        up, buy = outcome == "Up", side == "BUY"
        d = 1 if up == buy else -1                       # Up-space taker direction
        x = price if up else 1 - price
        notional = price * size
        while recent and recent[0][0] < t - 10:
            recent_sum -= recent.popleft()[1]
        k = max(t - DELAY, 0)
        rows.append((
            start, t, d, x, size, notional, y,
            bot.value(), *(flows[B].value() for B in FLOW_SIZES), sig50.signed(),
            len(recent), recent_sum,
            abs(fair[k] - fair[max(k - 10, 0)]),           # spot-model move over the 10 s before the match
            fair[k], fair[min(k + 5, E.WINDOW)], fair[min(k + 30, E.WINDOW)],
        ))
        # update trackers after recording the pre-trade state
        if up:
            bot.add(notional, buy)
        for B in FLOW_SIZES:
            flows[B].add(notional, d == 1)
        recent.append((t, notional)); recent_sum += notional
        marks.append((t, bot.value()))
    path_bot = np.full(E.WINDOW + 1, np.nan)
    for t, v in marks:                                   # value known at the end of each second
        path_bot[t:] = v
    return rows, (start, path_bot)


COLS = ["start", "t", "dir", "x", "size", "notional", "y",
        "vpin_bot", "vpin_flow5", "vpin_flow50", "vpin_flow500", "signed50",
        "n_10s", "notional_10s", "spot_move_10s", "f_fill", "f_5", "f_30"]


def build():
    m = pd.read_parquet("data/markets.parquet")
    m = m[m.outcome_up.isin(["0", "1"]) & (m.outcome_up != m.outcome_down)]
    y = dict(zip(m.start, (m.outcome_up == "1").astype(int)))
    fair = {int(k): v for k, v in np.load("data/fair.npz").items()}
    t = pd.read_parquet("data/trades.parquet")
    t = t[t.start.isin(fair.keys()) & t.start.isin(y.keys())].copy()
    t["t"] = t.timestamp - t.start
    # the API lists newest first: reverse within each market, then stable sort by time
    t["seq"] = -np.arange(len(t))
    t = t[(t.t >= 0) & (t.t < E.WINDOW)].sort_values(["start", "t", "seq"], kind="stable")
    jobs = [(s, g[["t", "outcome", "side", "price", "size"]], fair[s], y[s]) for s, g in t.groupby("start")]
    with Pool(8) as p:
        out = p.map(market_features, jobs, chunksize=50)
    feats = pd.DataFrame([r for rows, _ in out for r in rows], columns=COLS)
    paths = {s: pth for _, (s, pth) in out}
    feats.to_parquet("data/vpin_features.parquet", index=False)
    np.savez("data/vpin_bot_paths.npz", **{str(k): v for k, v in paths.items()})
    return feats, paths


if __name__ == "__main__":
    feats, _ = build()
    print(len(feats), "trades with features")
    print(feats.describe().T.round(4).to_string())
