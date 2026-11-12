"""Backtester. Key honesty rules:
- NO lookahead: a position decided at today's close earns TOMORROW's return (shift by 1).
- Costs: every change in position pays `cost_bps` (fees + spread + slippage) on the amount traded."""
import numpy as np
import pandas as pd


def backtest(px: pd.Series, position: pd.Series, cost_bps: float = 10.0) -> pd.DataFrame:
    ret = px.pct_change().fillna(0.0)
    pos = position.shift(1).fillna(0.0)          # trade at the close, earn the next day's move
    turnover = pos.diff().abs().fillna(pos.abs())
    strat = pos * ret - turnover * cost_bps / 10_000
    return pd.DataFrame({"ret": strat, "equity": (1 + strat).cumprod(), "pos": pos,
                         "turnover": turnover})


def metrics(bt: pd.DataFrame) -> dict:
    r = bt["ret"]
    years = len(r) / 252
    eq = bt["equity"]
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252) / vol if vol > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    trades = int((bt["turnover"] > 0).sum())
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": dd, "Trades": trades,
            "TimeInvested": bt["pos"].mean()}
