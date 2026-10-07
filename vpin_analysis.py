"""Analysis for the VPIN article. Needs data/vpin_features.parquet from vpin_study.py."""
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

import engine as E

pd.set_option("display.width", 220)
OUT = {}
rng = np.random.default_rng(0)


def load():
    f = pd.read_parquet("data/vpin_features.parquet")
    q, n = f["size"], f.notional
    f["spread"] = f.dir * (f.x - f.f_fill) * q          # maker vs fair at the match
    f["pnl30"] = f.dir * (f.x - f.f_30) * q
    f["pnlres"] = f.dir * (f.x - f.y) * q
    f["tox30"] = f.dir * (f.f_30 - f.f_fill) * q        # fair move in the taker's favour (maker's loss)
    f["fut30"] = f.f_30 - f.f_fill                       # signed future move, Up space
    return f


def bp(g, col):
    return g[col].sum() / g.notional.sum() * 1e4


def boot_diff(f, mask_hi, mask_lo, col, n=500):
    """Market-cluster bootstrap of bp(col | hi) - bp(col | lo)."""
    keys = f.start.to_numpy()
    agg = pd.DataFrame({"start": keys, "hi_v": np.where(mask_hi, f[col], 0), "hi_n": np.where(mask_hi, f.notional, 0),
                        "lo_v": np.where(mask_lo, f[col], 0), "lo_n": np.where(mask_lo, f.notional, 0)}).groupby("start").sum().to_numpy()
    idx = rng.integers(0, len(agg), (n, len(agg)))
    d = []
    for i in idx:
        s = agg[i].sum(0)
        d.append((s[0] / s[1] - s[2] / s[3]) * 1e4)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def deciles(f, col, label):
    g = f.dropna(subset=[col]).copy()
    g["dec"] = pd.qcut(g[col].rank(method="first"), 10, labels=range(1, 11))
    rows = []
    for d, h in g.groupby("dec", observed=True):
        rows.append({"decile": int(d), "vpin": h[col].mean(), "notional_share": h.notional.sum() / g.notional.sum(),
                     "spread_bp": bp(h, "spread"), "tox30_bp": bp(h, "tox30"), "pnl30_bp": bp(h, "pnl30"),
                     "pnlres_bp": bp(h, "pnlres"), "trade_usd": h.notional.mean(), "trades_10s": h.n_10s.mean(),
                     "spot_move_10s_c": h.spot_move_10s.mean() * 100})
    t = pd.DataFrame(rows)
    hi, lo = g.dec == 10, g.dec == 1
    ci = boot_diff(g, hi.to_numpy(), lo.to_numpy(), "tox30")
    print(f"\n=== deciles of {label} ({len(g):,} trades with a value) ===")
    print(t.round(3).to_string(index=False))
    print(f"top minus bottom decile, tox30: {bp(g[hi], 'tox30') - bp(g[lo], 'tox30'):.1f} bp, 95% CI {ci[0]:.1f} to {ci[1]:.1f}")
    OUT[f"deciles_{col}"] = t.round(5).to_dict(orient="records")
    OUT[f"top_minus_bottom_{col}"] = {"tox30_bp": bp(g[hi], "tox30") - bp(g[lo], "tox30"), "ci": ci}


def regression(f, n_boot=200):
    """Per-share toxicity (cents) on standardized features, notional-weighted, market-cluster bootstrap."""
    g = f.dropna(subset=["vpin_bot", "vpin_flow50"]).copy()
    p = g.x
    X = pd.DataFrame({
        "vpin_bot": g.vpin_bot, "vpin_flow50": g.vpin_flow50,
        "log_trades_10s": np.log1p(g.n_10s), "spot_move_10s": g.spot_move_10s,
        "p_one_minus_p": p * (1 - p), "time_in_window": g.t / E.WINDOW, "log_trade_usd": np.log(g.notional.clip(0.01)),
    })
    X = (X - X.mean()) / X.std()
    yv = (g.tox30 / g["size"] * 100).to_numpy()          # cents per share
    w = g.notional.to_numpy()
    starts = g.start.to_numpy()

    def fit(cols, rows=None):
        A = np.column_stack([np.ones(len(X))] + [X[c].to_numpy() for c in cols])
        if rows is not None:
            A, yy, ww = A[rows], yv[rows], w[rows]
        else:
            yy, ww = yv, w
        sw = np.sqrt(ww)
        beta, *_ = np.linalg.lstsq(A * sw[:, None], yy * sw, rcond=None)
        resid = yy - A @ beta
        r2 = 1 - np.sum(ww * resid ** 2) / np.sum(ww * (yy - np.average(yy, weights=ww)) ** 2)
        return beta, r2

    specs = {"vpin_bot alone": ["vpin_bot"], "vpin_flow50 alone": ["vpin_flow50"],
             "spot move alone": ["spot_move_10s"],
             "all": list(X.columns)}
    uniq = np.unique(starts)
    pos = pd.Series(np.arange(len(starts))).groupby(starts).apply(lambda s: s.to_numpy())
    res = {}
    for name, cols in specs.items():
        beta, r2 = fit(cols)
        bs = []
        for _ in range(n_boot):
            pick = rng.choice(uniq, len(uniq))
            rows = np.concatenate([pos[s] for s in pick])
            bs.append(fit(cols, rows)[0])
        bs = np.array(bs)
        res[name] = {"r2": r2, "coef": {c: (float(beta[i + 1]), float(np.percentile(bs[:, i + 1], 2.5)),
                                            float(np.percentile(bs[:, i + 1], 97.5))) for i, c in enumerate(cols)}}
        print(f"\n--- {name}: R2 {r2:.4f}")
        for c, (b, lo, hi) in res[name]["coef"].items():
            print(f"  {c:16s} {b:+.4f} c/share per 1 sd   95% CI {lo:+.4f} to {hi:+.4f}")
    OUT["regression"] = res


def signed(f):
    g = f.dropna(subset=["signed50"]).copy()
    g["dec"] = pd.qcut(g.signed50.rank(method="first"), 10, labels=range(1, 11))
    t = g.groupby("dec", observed=True).agg(signed=("signed50", "mean"), fut30_c=("fut30", lambda s: s.mean() * 100))
    c = np.corrcoef(g.signed50, g.fut30)[0, 1]
    print(f"\n=== signed VPIN ($50) vs next-30s fair move: corr {c:.4f}")
    print(t.round(4).to_string())
    OUT["signed"] = {"corr": c, "deciles": t.reset_index().astype({"dec": int}).round(5).to_dict(orient="records")}


# ---------------------------------------------------------------- quoter with pauses
def _sim(args):
    cfg, items = args
    pnl = reb = vol = fills = 0.0
    per_market = []
    for start, tr, fair, y, pause in items:
        p, r, fl = E.simulate_market(tr, fair, y, center="spot", half=cfg["half"], latency=3, through=0.0,
                                     pause=pause, pause_mode=cfg["mode"])
        pnl += p; reb += r; fills += len(fl)
        vol += sum(f[2] * f[3] if f[1] == 1 else (1 - f[2]) * f[3] for f in fl)
        per_market.append((start, p + r))
    return cfg, pnl, reb, vol, fills, per_market


def flow500_paths(keys):
    """Per-second flow VPIN ($500 buckets): last pre-trade value seen up to each second."""
    f = pd.read_parquet("data/vpin_features.parquet", columns=["start", "t", "vpin_flow500"])
    f = f[f.start.isin(set(keys))]
    out = {}
    for s, g in f.groupby("start"):
        p = np.full(E.WINDOW + 1, np.nan)
        for t, v in zip(g.t.to_numpy(), g.vpin_flow500.to_numpy()):
            p[t:] = v
        out[s] = p
    return out


def quoter_with_pauses():
    mk, tr = E.load_markets()
    fair = {int(k): v for k, v in np.load("data/fair.npz").items()}
    vp = {int(k): v for k, v in np.load("data/vpin_bot_paths.npz").items()}
    w = tr[(tr.t >= 0) & (tr.t < E.WINDOW) & tr.start.isin(list(vp))]
    w = w.sort_values(["start", "t"], kind="stable")      # simulate_market expects time order (same as run.py)
    groups = {s: g[["t", "dir", "x", "size"]] for s, g in w.groupby("start")}
    keys = sorted(groups)
    f500 = flow500_paths(keys)

    def spot_move(s):
        f = fair[s]
        m = np.zeros(E.WINDOW + 1)
        m[10:] = np.abs(f[10:] - f[:-10])
        return m

    signals = {"bot VPIN": {s: np.nan_to_num(vp[s]) for s in keys},
               "flow VPIN $500": {s: np.nan_to_num(f500.get(s, np.zeros(E.WINDOW + 1))) for s in keys},
               "spot move (10 s)": {s: spot_move(s) for s in keys}}
    rules = {"no pause": ({s: None for s in keys}, 0.0)}
    for thr in (0.7, 0.85):
        frac = float(np.mean(np.concatenate([signals["bot VPIN"][s][:E.WINDOW - 15] for s in keys]) > thr))
        rules[f"bot rule: VPIN > {thr}"] = ({s: signals["bot VPIN"][s] > thr for s in keys}, frac)
    share = 0.20
    jit = np.random.default_rng(42)
    for name, sig in signals.items():
        # tiny jitter breaks ties (bot VPIN sits at exactly 1.0 much of the time) so each rule pauses 20%
        jsig = {s: sig[s] + jit.random(E.WINDOW + 1) * 1e-9 for s in keys}
        thr = np.quantile(np.concatenate([jsig[s][:E.WINDOW - 15] for s in keys]), 1 - share)
        rules[f"top 20% {name}"] = ({s: jsig[s] > thr for s in keys}, share)
    for seed in range(3):
        r = np.random.default_rng(seed)
        rules[f"random 20% #{seed}"] = ({s: r.random(E.WINDOW + 1) < share for s in keys}, share)
    for thr in (0.7, 0.85):
        frac = rules[f"bot rule: VPIN > {thr}"][1]
        for seed in range(3):
            r = np.random.default_rng(100 + seed)
            rules[f"random {frac:.0%} (placebo for VPIN > {thr}) #{seed}"] = ({s: r.random(E.WINDOW + 1) < frac for s in keys}, frac)
    jobs = []
    chunks = np.array_split(np.arange(len(keys)), 8)
    for half in (0.03, 0.05):
        for mode in ("all", "entry"):
            for name, (paths, frac) in rules.items():
                if mode == "entry" and name == "no pause":
                    continue
                for c in chunks:
                    items = [(keys[i], groups[keys[i]], fair[keys[i]], int(mk.loc[keys[i], "y"]), paths[keys[i]]) for i in c]
                    jobs.append(({"half": half, "mode": mode, "rule": name, "pause_share": frac}, items))
    with Pool(8) as p:
        out = p.map(_sim, jobs)
    agg = {}
    for cfg, pnl, reb, vol, fills, pm in out:
        k = (cfg["half"], cfg["mode"], cfg["rule"])
        a_ = agg.setdefault(k, dict(cfg, total=0.0, volume=0.0, fills=0, pm=[]))
        a_["total"] += pnl + reb; a_["volume"] += vol; a_["fills"] += fills; a_["pm"] += pm
    rows = []
    brng = np.random.default_rng(7)
    for a_ in agg.values():
        pm = pd.Series(dict(a_.pop("pm")))
        daily = pm.groupby(pd.to_datetime(pm.index, unit="s").date).sum().to_numpy()
        bs = [daily[brng.integers(0, len(daily), len(daily))].sum() for _ in range(2000)]
        a_["ci_lo"], a_["ci_hi"] = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        rows.append(a_)
    df = pd.DataFrame(rows)
    df["rule_group"] = df.rule.str.replace(r" #\d$", "", regex=True)
    df = df.groupby(["half", "mode", "rule_group"], as_index=False).agg(
        pause_share=("pause_share", "first"), total=("total", "mean"), volume=("volume", "mean"),
        fills=("fills", "mean"), ci_lo=("ci_lo", "mean"), ci_hi=("ci_hi", "mean"))
    df["bp"] = df.total / df.volume * 1e4
    df["per_day"] = df.total / 30
    print(f"\n=== spot quoter, real latency ~0.8 s, fill at my price, {len(keys)} markets: effect of pausing ===")
    print(df.round(2).to_string(index=False))
    OUT["quoter_pauses"] = df.round(4).to_dict(orient="records")


if __name__ == "__main__":
    import sys
    if "--quoter-only" in sys.argv:
        OUT.update(json.load(open("vpin_results.json")))
        quoter_with_pauses()
        json.dump(OUT, open("vpin_results.json", "w"), indent=1, default=float)
        sys.exit()
    f = load()
    print(f"{len(f):,} in-window trades, ${f.notional.sum():,.0f}")
    for col in ["vpin_bot", "vpin_flow5", "vpin_flow50", "vpin_flow500"]:
        v = f[col]
        share = {thr: float((v > thr).mean()) for thr in (0.3, 0.7, 0.85)}
        print(f"{col:13s} missing {v.isna().mean():.3f}  median {v.median():.3f}  share >0.3/0.7/0.85: "
              + " / ".join(f"{share[t]:.3f}" for t in share))
        OUT[f"dist_{col}"] = {"missing": float(v.isna().mean()), "median": float(v.median()), "share_above": share}
    for col, label in [("vpin_bot", "bot VPIN (Up token, $5 x 20)"), ("vpin_flow50", "flow VPIN ($50 x 20)"),
                       ("vpin_flow500", "flow VPIN ($500 x 20)")]:
        deciles(f, col, label)
    print("\n=== Spearman correlations ===")
    cr = f[["vpin_bot", "vpin_flow5", "vpin_flow50", "vpin_flow500", "notional", "n_10s", "spot_move_10s"]].corr("spearman")
    print(cr.round(3).to_string())
    OUT["spearman"] = cr.round(4).to_dict()
    regression(f)
    signed(f)
    quoter_with_pauses()
    json.dump(OUT, open("vpin_results.json", "w"), indent=1, default=float)
