"""Charts for the VPIN article (dark site theme, validated palette)."""
import json
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "site-article-vpin/public/images/vpin"
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
r = json.load(open("vpin_results.json"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=200)
    plt.close(fig)


def label(ax, bars, fmt):
    for b in bars:
        ax.annotate(fmt(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 5),
                    textcoords="offset points", ha="center", color=FG, fontsize=11, fontweight="bold")


# 1. saturation by bucket size
cols = [("vpin_bot", "My bot\n(Up token, $5)"), ("vpin_flow5", "Both tokens\n$5 buckets"),
        ("vpin_flow50", "Both tokens\n$50 buckets"), ("vpin_flow500", "Both tokens\n$500 buckets")]
share = [r[f"dist_{c}"]["share_above"]["0.7"] * 100 for c, _ in cols]
med = [r[f"dist_{c}"]["median"] for c, _ in cols]
fig, ax = plt.subplots(figsize=(10, 4.6))
bars = ax.bar(range(4), share, 0.55, color=[BLUE, AQUA, AQUA, AQUA])
label(ax, bars, lambda v: f"{v:.0f}%")
ax.set_xticks(range(4), [f"{l}\nmedian {m:.2f}" for (_, l), m in zip(cols, med)])
ax.set_ylim(0, 110)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.set_ylabel("Trades with VPIN above 0.7", color=MUTED)
ax.grid(axis="x", visible=False)
ax.set_title("Share of trades above my bot's pause threshold, by bucket size", loc="left", color=FG, fontsize=13, pad=14)
save(fig, "saturation")

# 2. toxicity by decile
fig, ax = plt.subplots(figsize=(10, 4.8))
for col, c, lab in [("vpin_bot", BLUE, "My bot's VPIN (Up token, $5 buckets)"),
                    ("vpin_flow50", AQUA, "Both tokens, $50 buckets"),
                    ("vpin_flow500", ORANGE, "Both tokens, $500 buckets")]:
    d = pd.DataFrame(r[f"deciles_{col}"])
    ax.plot(d.decile, d.tox30_bp, color=c, lw=2, marker="o", ms=6, mec=BG, label=lab)
    ax.annotate(f"{d.tox30_bp.iloc[-1]:.0f}", (10, d.tox30_bp.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", color=FG, fontsize=10, fontweight="bold")
ax.set_xticks(range(1, 11), ["1\nlowest"] + [str(i) for i in range(2, 10)] + ["10\nhighest"])
ax.set_xlabel("VPIN decile, measured before the trade", color=MUTED)
ax.set_ylabel("Fair-value move against the maker\nover the next 30 s, bp", color=MUTED)
ax.set_xlim(0.6, 10.7)
ax.set_ylim(30, 132)
ax.legend(loc="upper center", fontsize=10, ncol=1)
ax.set_title("Adverse selection by VPIN decile: the sign depends on the bucket size", loc="left", color=FG, fontsize=13, pad=14)
save(fig, "deciles")

# 3. what predicts toxicity (standardised coefficients, market-cluster bootstrap)
coef = r["regression"]["all"]["coef"]
names = {"time_in_window": "Later in the window", "p_one_minus_p": "Price near 50/50", "vpin_bot": "My bot's VPIN",
         "spot_move_10s": "BTC move, last 10 s", "vpin_flow50": "VPIN, both tokens $50",
         "log_trades_10s": "Trades, last 10 s", "log_trade_usd": "Trade size"}
order = sorted(coef, key=lambda k: coef[k][0])
fig, ax = plt.subplots(figsize=(10, 4.8))
for i, k in enumerate(order):
    b, lo, hi = coef[k]
    sig = lo > 0 or hi < 0
    col = BLUE if sig else MUTED
    ax.plot([lo, hi], [i, i], color=col, lw=2)
    ax.plot([b], [i], "o", color=col, ms=8, mec=BG, mew=1.5)
ax.axvline(0, color=FG, lw=0.8, alpha=0.6)
ax.set_yticks(range(len(order)), [names[k] for k in order])
ax.set_xlabel("Effect of +1 standard deviation on the 30 s adverse move (cents per share)", color=MUTED)
ax.grid(axis="y", visible=False)
ax.set_title("What predicts toxic fills (grey: 95% interval includes zero)", loc="left", color=FG, fontsize=13, pad=14)
save(fig, "drivers")

# 4. filter vs matched random placebo (entry-side pauses, ±5c)
q = pd.DataFrame(r["quoter_pauses"])
q = q[(q.half == 0.05) & (q["mode"] == "entry")].set_index("rule_group")
pairs = [("top 20% bot VPIN", "random 20%", "Pause 20% of the time\n(top 20% VPIN)"),
         ("bot rule: VPIN > 0.85", "random 87% (placebo for VPIN > 0.85)", "Pause 87%\n(VPIN > 0.85)"),
         ("bot rule: VPIN > 0.7", "random 99% (placebo for VPIN > 0.7)", "Pause 99%\n(VPIN > 0.7)")]
fig, ax = plt.subplots(figsize=(10, 4.8))
w = 0.36
for j, (rule, plc, lab) in enumerate(pairs):
    for off, key, col in [(-w / 2 - 0.01, rule, BLUE), (w / 2 + 0.01, plc, ORANGE)]:
        row = q.loc[key]
        ax.bar(j + off, row.total / 1000, w, color=col)
        ax.plot([j + off] * 2, [row.ci_lo / 1000, row.ci_hi / 1000], color=FG, lw=1.2)
        y = row.total / 1000
        ax.annotate(f"{'+' if y > 0 else '−'}${abs(y):.1f}k", (j + off, min(y, row.ci_lo / 1000)), xytext=(0, -14),
                    textcoords="offset points", ha="center", color=FG, fontsize=10, fontweight="bold")
from matplotlib.patches import Patch
handles = [Patch(color=BLUE, label="Pause new positions when VPIN is high"),
           Patch(color=ORANGE, label="Pause new positions at random, same share of time")]
ax.axhline(0, color=FG, lw=0.8, alpha=0.6)
ax.set_xticks(range(3), [p[2] for p in pairs])
ax.yaxis.set_major_formatter(lambda v, _: f"${v:.0f}k".replace("$-", "−$"))
ax.set_ylabel("Quoter P&L over 30 days", color=MUTED)
ax.set_ylim(-48, 8)
ax.grid(axis="x", visible=False)
ax.legend(handles=handles, loc="lower right", fontsize=10)
ax.set_title("VPIN as a filter vs a random pause of the same length (±5¢, ~0.8 s latency)", loc="left", color=FG, fontsize=13, pad=14)
save(fig, "filter")
print("ok")
