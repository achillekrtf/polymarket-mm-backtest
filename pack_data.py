"""Pack the per-market JSON downloads into two parquet files for the release.

data/markets/*.json (1.9 GB) -> data/markets.parquet + data/trades.parquet
Trader wallet addresses are dropped; nothing in the analysis uses them."""
import glob
import json

import pandas as pd

mk, tr = [], []
for f in sorted(glob.glob("data/markets/*.json")):
    d = json.load(open(f))
    mk.append({"start": d["start"], "condition_id": d["condition_id"],
               "outcome_up": d["outcome_prices"][0], "outcome_down": d["outcome_prices"][1],
               "fee_rate": (d.get("fee") or {}).get("rate"), "rebate_rate": (d.get("fee") or {}).get("rebateRate")})
    for t in d["trades"]:
        tr.append((d["start"], t["timestamp"], t["side"], t["outcome"], t["price"], t["size"]))
pd.DataFrame(mk).to_parquet("data/markets.parquet", index=False)
trades = pd.DataFrame(tr, columns=["start", "timestamp", "side", "outcome", "price", "size"])
trades = trades.astype({"side": "category", "outcome": "category"})
trades.to_parquet("data/trades.parquet", index=False, compression="zstd")
print(len(mk), "markets,", len(trades), "trades")
