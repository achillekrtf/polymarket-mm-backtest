"""Download every BTC 5-minute Up/Down market in a date range from Polymarket's public APIs.

For each window start T (every 300 s) the market slug is btc-updown-5m-<T>.
  Gamma API  -> conditionId, resolution (outcomePrices), fee schedule
  Data API   -> every taker trade (side, outcome, price, size, timestamp)
One JSON file per market in data/markets/, so the download can be resumed.
"""
import json
import subprocess
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

OUT = Path("data/markets")


def get(url, retries=8, want_list=False):
    """GET JSON. With want_list, anything other than a JSON list (rate-limit page, error
    object) is retried with backoff, so an empty list always means 'no more data'."""
    for i in range(retries):
        try:
            r = json.loads(subprocess.check_output(["curl", "-s", "--max-time", "20", url]))
            if not want_list or isinstance(r, list):
                return r
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
        time.sleep(2 ** min(i, 5))
    raise RuntimeError(f"failed after retries: {url}")


def fetch(start_ts):
    path = OUT / f"{start_ts}.json"
    if path.exists():
        return "cached"
    try:
        ev = get(f"https://gamma-api.polymarket.com/events?slug=btc-updown-5m-{start_ts}", want_list=True)
        if not ev:
            return "missing"
    except RuntimeError:
        return "error"
    if not ev:
        return "missing"
    m = ev[0]["markets"][0]
    trades, off = [], 0
    while True:
        try:
            page = get(f"https://data-api.polymarket.com/trades?market={m['conditionId']}&limit=500&offset={off}&takerOnly=true", want_list=True)
        except RuntimeError:
            return "error"   # nothing written: the market is retried on the next run
        if not page:
            break
        trades += [{k: t[k] for k in ("timestamp", "side", "outcome", "price", "size", "proxyWallet")} for t in page]
        off += 500
    rec = {
        "start": start_ts,
        "condition_id": m["conditionId"],
        "outcome_prices": json.loads(m["outcomePrices"]),
        "fee": m.get("feeSchedule"),
        "volume": m.get("volume"),
        "trades": trades,
    }
    path.write_text(json.dumps(rec))
    return len(trades)


if __name__ == "__main__":
    start, end = sys.argv[1], sys.argv[2]
    OUT.mkdir(parents=True, exist_ok=True)
    ts = range(int(pd.Timestamp(start, tz="UTC").timestamp()), int(pd.Timestamp(end, tz="UTC").timestamp()), 300)
    done = 0
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(fetch, ts):
            done += 1
            if done % 200 == 0:
                print(done, "markets", r, flush=True)
    print("finished", done)
