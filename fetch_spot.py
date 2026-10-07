"""Binance BTCUSDT 1-second klines (close) for the backtest period, saved as parquet per day."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

OUT = Path("data/spot")


def day(d):
    path = OUT / f"{d.date()}.parquet"
    if path.exists():
        return
    start = int(d.timestamp() * 1000)
    end = start + 86_400_000
    rows = []
    while start < end:
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1s&startTime={start}&endTime={end - 1}&limit=1000"
        for _ in range(5):
            try:
                batch = json.loads(subprocess.check_output(["curl", "-s", "--max-time", "20", url]))
                if isinstance(batch, list):
                    break
            except Exception:
                pass
        if not batch:
            break
        rows += [(b[0] // 1000, float(b[4])) for b in batch]
        start = batch[-1][0] + 1000
    pd.DataFrame(rows, columns=["ts", "close"]).to_parquet(path)
    print(d.date(), len(rows), flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(sys.argv[1], sys.argv[2], freq="D", tz="UTC", inclusive="left")
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(day, days))
