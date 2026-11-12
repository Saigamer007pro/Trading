"""Usage:
  python run_backtest.py --ticker SPY          # real data (needs internet)
  python run_backtest.py --synthetic           # fake data, only to test the code
Splits history in two: the first part to look at, the LAST 30% kept as an untouched 'out-of-sample' test."""
import argparse
import pandas as pd
from data import load_prices, synthetic_prices
from strategies import STRATEGIES
from backtest import backtest, metrics

ap = argparse.ArgumentParser()
ap.add_argument("--ticker", default="SPY")
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--cost-bps", type=float, default=10.0)
a = ap.parse_args()

px = synthetic_prices() if a.synthetic else load_prices(a.ticker)
split = px.index[int(len(px) * 0.7)]
print(f"{'SYNTHETIC' if a.synthetic else a.ticker}: {px.index[0].date()} -> {px.index[-1].date()}, "
      f"out-of-sample starts {split.date()}, cost {a.cost_bps} bps/trade\n")

rows = []
for name, fn in STRATEGIES.items():
    bt = backtest(px, fn(px), a.cost_bps)       # signals computed on full history (uses past only)
    for label, part in (("in-sample", bt[bt.index < split]), ("OUT-OF-SAMPLE", bt[bt.index >= split])):
        part = part.copy(); part["equity"] = (1 + part["ret"]).cumprod()
        rows.append({"strategy": name, "period": label, **metrics(part)})
df = pd.DataFrame(rows)
fmt = {c: "{:.1%}" for c in ["CAGR", "Vol", "MaxDD", "TimeInvested"]}
fmt["Sharpe"] = "{:.2f}"
print(df.to_string(index=False, formatters={k: v.format for k, v in fmt.items()}))
