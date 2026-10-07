import json
import sys
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import engine as E

pd.set_option("display.width", 220)
OUT = {}


def build():
    mk, tr = E.load_markets()
    spot, vol = E.load_spot()
    cache = Path("data/fair.npz")
    fair = dict(np.load(cache)) if cache.exists() else {}
    for s in mk.index:
        if str(s) not in fair:
            p = E.fair_path(spot, vol, s, E.RULE)
            if p is not None:
                fair[str(s)] = p
    np.savez(cache, **fair)
    keep = [s for s in mk.index if str(s) in fair]
    mk = mk.loc[keep]
    tr = tr[tr.start.isin(keep)].sort_values(["start", "t"], kind="stable").reset_index(drop=True)
    return mk, tr, {int(k): v for k, v in fair.items()}, spot


# ---------------------------------------------------------------- A. what makers earned
def maker_economics(mk, tr, fair):
    tr = tr.copy()
    tr["maker_pnl"] = tr.dir * (tr.x - tr.y) * tr["size"]
    tr["taker_fee"] = E.FEE_RATE * tr.x * (1 - tr.x) * tr["size"]
    tr["rebate"] = E.REBATE_SHARE * tr.taker_fee
    bins = [-10**7, -600, -60, 0, 60, 120, 180, 240, 300, 10**7]
    labels = ["> 10 min before", "10 to 1 min before", "last minute before", "0-60s", "60-120s",
              "120-180s", "180-240s", "240-300s", "after close"]
    tr["bucket"] = pd.cut(tr.t, bins, right=False, labels=labels)
    g = tr.groupby("bucket", observed=True).agg(notional=("notional", "sum"), trades=("x", "size"),
                                                maker_pnl=("maker_pnl", "sum"), fees=("taker_fee", "sum"),
                                                rebate=("rebate", "sum"))
    g["share_of_volume"] = g.notional / g.notional.sum()
    g["maker_bps"] = g.maker_pnl / g.notional * 1e4
    g["maker_bps_incl_rebate"] = (g.maker_pnl + g.rebate) / g.notional * 1e4
    g["taker_fee_bps"] = g.fees / g.notional * 1e4
    print("\n=== A. Maker P&L to resolution, by time to window ===")
    print(g.round(2).to_string())
    tot = g.sum(numeric_only=True)
    print(f"TOTAL notional ${tot.notional:,.0f}  trades {tot.trades:,.0f}  maker pnl ${tot.maker_pnl:,.0f} "
          f"({tot.maker_pnl / tot.notional * 1e4:.1f} bp)  taker fees ${tot.fees:,.0f}  rebates ${tot.rebate:,.0f}")

    # decomposition for in-window trades: spread vs fair, then fair -> outcome
    w = tr[(tr.t >= 0) & (tr.t < E.WINDOW)].copy()
    ts = w.t.to_numpy()
    fs = np.array([fair[s] for s in w.start])
    f0 = fs[np.arange(len(w)), ts]
    f5 = fs[np.arange(len(w)), np.minimum(ts + 5, E.WINDOW)]
    f30 = fs[np.arange(len(w)), np.minimum(ts + 30, E.WINDOW)]
    w["f0"] = f0
    d, x, q, y = w.dir.to_numpy(), w.x.to_numpy(), w["size"].to_numpy(), w.y.to_numpy()
    dec = {
        "spread_vs_fair_at_fill": (d * (x - f0) * q).sum(),
        "markout_5s": (d * (x - f5) * q).sum(),
        "markout_30s": (d * (x - f30) * q).sum(),
        "to_resolution": (d * (x - y) * q).sum(),
    }
    notional = w.notional.sum()
    print("\n=== A2. In-window maker P&L vs fair value (bp of notional) ===")
    for k, v in dec.items():
        print(f"{k:24s} ${v:12,.0f}  {v / notional * 1e4:7.1f} bp")
    by_price = w.assign(pb=pd.cut(w.x, [0, .1, .3, .5, .7, .9, 1])).groupby("pb", observed=True).apply(
        lambda g: pd.Series({"notional": g.notional.sum(),
                             "maker_bps": (g.dir * (g.x - g.y) * g["size"]).sum() / g.notional.sum() * 1e4}),
        include_groups=False)
    print("\nby Up-space price:\n", by_price.round(1).to_string())

    # C. calibration: market price vs model, in-window trades
    brier_mkt = np.mean((x - y) ** 2)
    brier_model = np.mean((f0 - y) ** 2)
    cal = w.assign(b=pd.cut(w.x, np.linspace(0, 1, 11))).groupby("b", observed=True).agg(
        price=("x", "mean"), up_rate=("y", "mean"), model=("f0", "mean"), n=("x", "size"))
    print(f"\n=== C. Brier score, in-window trades: market price {brier_mkt:.4f}  model {brier_model:.4f}")
    print(cal.round(3).to_string())

    # cluster bootstrap over markets (trades inside one market share the same outcome)
    rng = np.random.default_rng(0)
    pm = w.assign(res=d * (x - y) * q, spr=d * (x - f0) * q, m30=d * (x - f30) * q,
                  reb=E.REBATE_SHARE * E.FEE_RATE * x * (1 - x) * q).groupby("start")[["res", "spr", "m30", "reb", "notional"]].sum()
    idx = rng.integers(0, len(pm), (2000, len(pm)))
    arr = pm.to_numpy()
    boot = {c: [] for c in ["res", "spr", "m30", "res_reb"]}
    for i in idx:
        s_ = arr[i].sum(0)
        boot["res"].append(s_[0] / s_[4] * 1e4); boot["spr"].append(s_[1] / s_[4] * 1e4)
        boot["m30"].append(s_[2] / s_[4] * 1e4); boot["res_reb"].append((s_[0] + s_[3]) / s_[4] * 1e4)
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boot.items()}
    print("\n95% CI (bootstrap over markets), bp:", {k: tuple(round(z, 1) for z in v) for k, v in ci.items()})
    OUT["ci_bp"] = ci
    OUT["maker_by_bucket"] = g.reset_index().astype({"bucket": str}).round(4).to_dict(orient="records")
    OUT["maker_total"] = {k: float(v) for k, v in tot.items()}
    OUT["decomposition_bp"] = {k: v / notional * 1e4 for k, v in dec.items()}
    OUT["decomposition_bp"]["to_resolution_plus_rebates"] = (dec["to_resolution"] + (E.REBATE_SHARE * E.FEE_RATE * x * (1 - x) * q).sum()) / notional * 1e4
    OUT["window_notional"] = notional
    OUT["brier"] = {"market": brier_mkt, "model": brier_model}
    OUT["calibration"] = cal.reset_index(drop=True).round(4).to_dict(orient="records")
    OUT["by_price"] = by_price.reset_index().astype({"pb": str}).round(2).to_dict(orient="records")


# ---------------------------------------------------------------- B. simulated quoter
def _run(args):
    params, items = args
    pnl = reb = vol = n = 0.0
    per_market = []
    for s, tr, fair, y in items:
        p, r, fills = E.simulate_market(tr, fair, y, **params)
        per_market.append((s, p + r))
        pnl += p; reb += r; n += len(fills)
        vol += sum(f[2] * f[3] if f[1] == 1 else (1 - f[2]) * f[3] for f in fills)
    return params, pnl, reb, vol, n, per_market


def quoting(mk, tr, fair):
    w = tr[(tr.t >= 0) & (tr.t < E.WINDOW)]
    items = [(s, g[["t", "dir", "x", "size"]], fair[s], int(mk.loc[s, "y"])) for s, g in w.groupby("start")]
    grid = [dict(center=c, half=h, latency=l, through=th)
            for c, h, l, th in product(["naive", "spot"], [0.01, 0.02, 0.03, 0.05], [1, 2, 3, 4, 5], [0.0, 0.01])]
    chunks = np.array_split(np.arange(len(items)), 8)
    jobs = [(p, [items[i] for i in c]) for p in grid for c in chunks]
    with Pool(8) as pool:
        res = pool.map(_run, jobs)
    agg = {}
    for params, pnl, reb, vol, n, pm in res:
        key = tuple(params.values())
        a = agg.setdefault(key, dict(params, pnl=0, rebate=0, volume=0, fills=0, per_market=[]))
        a["pnl"] += pnl; a["rebate"] += reb; a["volume"] += vol; a["fills"] += n; a["per_market"] += pm
    rows = []
    days = (mk.index.max() - mk.index.min()) / 86400
    for a in agg.values():
        pm = pd.Series(dict(a.pop("per_market"))).sort_index()
        daily = pm.groupby(pd.to_datetime(pm.index, unit="s").date).sum()
        a.update(total=a["pnl"] + a["rebate"], per_day=(a["pnl"] + a["rebate"]) / days,
                 bp=(a["pnl"] + a["rebate"]) / a["volume"] * 1e4 if a["volume"] else np.nan,
                 win_days=(daily > 0).mean(), sharpe=daily.mean() / daily.std() * np.sqrt(365) if daily.std() else np.nan,
                 markets_with_fills=(pm != 0).mean(),
                 daily={str(k): float(v) for k, v in daily.items()})
        rows.append(a)
    df = pd.DataFrame(rows).sort_values(["center", "latency", "through", "half"])
    print("\n=== B. Simulated quoter (20 shares per quote, max inventory 100 shares, held to resolution) ===")
    print(df.drop(columns=["daily"]).round(2).to_string(index=False))
    OUT["quoting"] = df.round(4).to_dict(orient="records")
    OUT["n_markets"] = len(items)
    OUT["days"] = days


if __name__ == "__main__":
    mk, tr, fair, spot = build()
    print(f"markets {len(mk)}, trades {len(tr):,}, {pd.to_datetime(mk.index.min(), unit='s')} -> "
          f"{pd.to_datetime(mk.index.max(), unit='s')}, Up rate {mk.y.mean():.3f}")
    OUT["period"] = [str(pd.to_datetime(mk.index.min(), unit="s")), str(pd.to_datetime(mk.index.max(), unit="s"))]
    OUT["up_rate"] = mk.y.mean()
    maker_economics(mk, tr, fair)
    if "--no-sim" not in sys.argv:
        quoting(mk, tr, fair)
    json.dump(OUT, open("results.json", "w"), indent=1, default=float)
