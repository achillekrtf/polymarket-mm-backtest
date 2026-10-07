"""Record Polymarket CLOB trade events (match time) for the next BTC 5m window.

Later compared with the Data API timestamps of the same trades to measure reporting delay."""
import asyncio
import json
import os
import ssl
import subprocess
import time

import websockets

# CERTS: optional CA bundle path, for networks behind a TLS-inspecting proxy
ctx = ssl.create_default_context(cafile=os.environ.get("CERTS"))
start = (int(time.time()) // 300 + 1) * 300
ev = json.loads(subprocess.check_output(["curl", "-s", f"https://gamma-api.polymarket.com/events?slug=btc-updown-5m-{start}"]))
m = ev[0]["markets"][0]
tokens = json.loads(m["clobTokenIds"])
print("window", start, m["conditionId"], flush=True)


async def main():
    out = open(f"data/ws_{start}.jsonl", "w")
    async with websockets.connect("wss://ws-subscriptions-clob.polymarket.com/ws/market", ssl=ctx, ping_interval=10) as ws:
        await ws.send(json.dumps({"assets_ids": tokens, "type": "market"}))
        while time.time() < start + 320:
            msg = await asyncio.wait_for(ws.recv(), timeout=30)
            now = time.time()
            for e in (json.loads(msg) if msg.startswith("[") else [json.loads(msg)]):
                if isinstance(e, dict) and e.get("event_type") == "last_trade_price":
                    e["recv"] = now
                    out.write(json.dumps(e) + "\n")
    out.close()
    json.dump({"start": start, "condition_id": m["conditionId"], "tokens": tokens}, open(f"data/ws_{start}_meta.json", "w"))
    print("done", flush=True)

asyncio.run(main())
