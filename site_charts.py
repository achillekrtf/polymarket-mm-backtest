"""Charts for achillekrtf.github.io (dark theme, validated palette)."""
import json
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "site-article/public/images/polymarket-mm"
os.makedirs(OUT, exist_ok=True)
BG, FG, MUTED, GRID, AXIS = "#191c1f", "#ececec", "#8b9098", "#2a2e33", "#3a3f45"
BLUE, ORANGE, AQUA, RED = "#3987e5", "#d95926", "#199e70", "#e66767"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "text.color": FG,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": AXIS, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.labelcolor": FG,
})
r = json.load(open("results.json"))
DELAY = json.load(open("data/ws_delay.json"))["median"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=200)
    plt.close(fig)


# 1. markout: what a maker earns vs fair value, as time passes after the fill
dec = r["decomposition_bp"]
ci = r["ci_bp"]
labels = ["At the fill", "5 s later", "30 s later", "At resolution", "Resolution\n+ rebates"]
vals = [dec["spread_vs_fair_at_fill"], dec["markout_5s"], dec["markout_30s"], dec["to_resolution"],
        dec["to_resolution_plus_rebates"]]
errs = [None, None, (ci["m30"][0], ci["m30"][1]), (ci["res"][0], ci["res"][1]), (ci["res_reb"][0], ci["res_reb"][1])]
fig, ax = plt.subplots(figsize=(10, 4.8))
colors = [BLUE] * 4 + [AQUA]
bars = ax.bar(range(5), vals, 0.56, color=colors)
for i, (b, v, e) in enumerate(zip(bars, vals, errs)):
    if e:
        ax.plot([i, i], e, color=FG, lw=1.2)
        ax.plot([i - 0.06, i + 0.06], [e[0]] * 2, color=FG, lw=1.2)
        ax.plot([i - 0.06, i + 0.06], [e[1]] * 2, color=FG, lw=1.2)
    top = max(v, e[1] if e else v)
    ax.annotate(f"{v:+.0f} bp" if abs(v) >= 10 else f"{v:+.1f} bp", (i, top), xytext=(0, 6),
                textcoords="offset points", ha="center", color=FG, fontsize=11, fontweight="bold")
ax.axhline(0, color=AXIS, lw=1)
ax.set_xticks(range(5), labels)
ax.set_ylabel("Maker P&L, bp of traded notional", color=MUTED)
ax.grid(axis="x", visible=False)
ax.set_title("Makers' edge vs the spot model, as time passes after the fill", loc="left", color=FG, fontsize=13, pad=14)
save(fig, "markout")

# 2. latency
q = pd.DataFrame(r["quoting"])
q["real"] = q.latency - DELAY
fig, ax = plt.subplots(figsize=(10, 5))
series = [("spot", 0.05, 0.0, BLUE, "Spot model, ±5¢, fill when a trade touches my price"),
          ("spot", 0.05, 0.01, AQUA, "Spot model, ±5¢, fill only when a trade goes through"),
          ("naive", 0.05, 0.0, ORANGE, "Follow the last trades, ±5¢")]
ax.axvspan(q.real.min() - 0.3, 0, color="#2b2f34", zorder=0)
ax.annotate("Shaded: the simulation\nsees spot moves from\nafter the trade", (q.real.min() - 0.2, -4300),
            color=MUTED, fontsize=9.5, va="top")
for c, h, th, col, lab in series:
    s = q[(q.center == c) & (q.half == h) & (q.through == th)].sort_values("real")
    ax.plot(s.real, s.per_day, color=col, lw=2, marker="o", ms=5, mec=BG, label=lab)
ax.axhline(0, color=FG, lw=0.8, alpha=0.6)
ax.set_xlabel("Real quote latency (seconds after the trade's match time)", color=MUTED)
ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}".replace("$-", "−$"))
ax.set_ylabel("P&L per day, incl. rebates", color=MUTED)
ax.set_xlim(q.real.min() - 0.3, q.real.max() + 0.3)
ax.legend(loc="upper right", fontsize=10)
ax.set_title("Simulated quoter: profit only exists before the trade happened", loc="left", color=FG, fontsize=13, pad=14)
save(fig, "latency")

# 3. lag correlation
lag = pd.read_csv("data/lag_corr.csv")
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.bar(lag.shift_s, lag["corr"], 0.7, color=[BLUE if s != -3 else AQUA for s in lag.shift_s])
ax.axvline(-DELAY, color=RED, lw=1.4, ls="--")
ax.annotate(f"Measured reporting delay\n({DELAY:.1f} s, 394 trades)", (-DELAY, 0.5), xytext=(10, 0), textcoords="offset points",
            color=FG, fontsize=10, va="center")
ax.set_xticks(lag.shift_s)
ax.set_xlabel("Shift applied to the spot model (seconds, negative = earlier)", color=MUTED)
ax.set_ylabel("Correlation of price changes", color=MUTED)
ax.grid(axis="x", visible=False)
ax.set_title("Trade prices follow BTC spot from 3 seconds before their timestamp", loc="left", color=FG, fontsize=13, pad=14)
save(fig, "timestamp-lag")
print("ok")
