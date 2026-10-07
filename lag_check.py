"""Estimate the delay between when a trade is matched and its reported timestamp.

If Polymarket prices react to BTC spot at match time, then price changes between two
trades line up best with spot-model changes shifted by -delay seconds."""
import numpy as np
import pandas as pd
import engine as E

mk, tr = E.load_markets()
fair = {int(k): v for k, v in np.load("data/fair.npz").items()}
w = tr[(tr.t >= 5) & (tr.t < 280) & tr.start.isin(fair.keys())].sort_values(["start", "t"], kind="stable")
# one price per market-second (median), then changes between consecutive seconds with trades
g = w.groupby(["start", "t"]).x.median().reset_index()
g = g[(g.x > 0.1) & (g.x < 0.9)]
g["dx"] = g.groupby("start").x.diff()
g["t0"] = g.groupby("start").t.shift()
g = g.dropna()
g = g[(g.t - g.t0) <= 3]
F = np.stack([fair[s] for s in g.start])
t1, t0 = g.t.to_numpy().astype(int), g.t0.to_numpy().astype(int)
rows = []
for k in range(-8, 6):
    a, b = np.clip(t1 + k, 0, E.WINDOW), np.clip(t0 + k, 0, E.WINDOW)
    df_ = F[np.arange(len(g)), a] - F[np.arange(len(g)), b]
    rows.append((k, np.corrcoef(g.dx, df_)[0, 1]))
df = pd.DataFrame(rows, columns=["shift_s", "corr"])
print(df.round(4).to_string(index=False))
df.to_csv("data/lag_corr.csv", index=False)
