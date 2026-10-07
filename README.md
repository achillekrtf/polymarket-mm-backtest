# Polymarket BTC 5-minute market making backtest

Can a two-sided quoter make money on Polymarket's "Bitcoin Up or Down" 5-minute markets? This repo measures what makers actually earned over 30 days (5 September to 4 October 2026, 8,640 markets, 12.9M taker trades) and simulates a quoter across spreads, fill rules and latencies.

Write-up: [Market making Polymarket's 5-minute Bitcoin markets](https://achillekrtf.github.io/articles/polymarket-btc-5m-market-making/)

## Headline results

| | |
|---|---|
| Taker fees paid | $3.86M on $205M of notional (188 bp) |
| Maker edge vs spot fair value at the fill | +108 bp |
| ... 30 seconds later | +57 bp |
| ... at resolution | +5.5 bp (95% CI −9 to +20) |
| ... at resolution, with rebates | +42 bp (27 to 56) |
| Brier score, traded price vs spot model | 0.162 vs 0.170 |
| Data API timestamp delay vs match time | 2.2 s on average (394 trades matched by tx hash) |
| Simulated quoter, realistic latency | Loses in every configuration |

Full output is in [`results.txt`](results.txt) and [`results.json`](results.json).

## Data

The packed dataset is attached to the [v1 release](../../releases/tag/v1) (80 MB):

```bash
gh release download v1 -R achillekrtf/polymarket-mm-backtest
tar -xzf polymarket-btc5m-2026-09-05_2026-10-04.tar.gz   # -> data/markets.parquet, data/trades.parquet, data/spot/
```

| File | Content |
|---|---|
| `data/markets.parquet` | One row per market: window start, condition id, outcome, fee and rebate rates (Gamma API) |
| `data/trades.parquet` | Every taker trade: window start, timestamp, side, outcome, price, size (Data API, wallets removed) |
| `data/spot/*.parquet` | Binance BTC/USDT 1-second closes |
| `data/ws_1791323100.jsonl` | One live market recorded from the CLOB websocket (match-time timestamps) |
| `data/ws_delay.json`, `data/lag_corr.csv` | Timestamp delay measurements |

Polymarket's Data API only keeps about a month of trades, so this period will not stay downloadable from the API. For a recent period, `fetch_markets.py` and `fetch_spot.py` download the raw files and `pack_data.py` packs them into the same parquet layout.

## Run it

```bash
python3 run.py            # maker economics, calibration, quoter grid -> results.json
python3 lag_check.py      # correlation of price changes with the spot model at each time shift
python3 site_charts.py    # charts for the article
```

Requires pandas, numpy, pyarrow and matplotlib. `run.py` builds `data/fair.npz` (the fair value path of every market) on the first run, which takes a few minutes.

## Method

- **Up space.** A trade on the Down token at price q is treated as the opposite trade on Up at 1 − q, so each market has one price.
- **Resolution rule.** Up if the 60-second TWAP at the end of the window is at or above the 60-second TWAP before it opens. With Binance prices this matches 97.5% of actual outcomes (other readings: 84 to 87%).
- **Fair value.** Probability that the last-minute average ends above the reference, with BTC as a random walk using the previous 30 minutes of realised 1-second volatility.
- **Maker economics.** The maker side of every taker trade, valued against the fair value at the fill, 5 s and 30 s later, and at resolution. Rebates are approximated as 20% of each trade's taker fee (0.07 × p × (1 − p) per share). Confidence intervals come from a bootstrap over markets.
- **Quoter.** Bid and ask around a center (spot fair value or an EWMA of trade prices), 20 shares per quote, inventory capped at 100 shares and held to resolution. A quote fills when a taker trade prints at (or, in the strict version, through) its price. Quotes at second s use information from second s − L.
- **Latency.** Data API timestamps are 2.2 s later than the match. A simulated latency L therefore corresponds to a real latency of about L − 2.2 s; the runs with L of 1 and 2 seconds use spot prices from after the trade.

## Limitations

One-second resolution, no queue position, Binance as a proxy for Chainlink, approximate rebate allocation, one month of data.

Personal research, not investment advice.
