"""Each strategy takes a price Series and returns a target position Series:
1.0 = fully invested, 0.0 = cash. Position at day t is decided using data up to the close of day t."""
import pandas as pd


def buy_and_hold(px: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=px.index)


def trend_sma(px: pd.Series, window: int = 200) -> pd.Series:
    """Invested while price is above its moving average, else cash."""
    return (px > px.rolling(window).mean()).astype(float)


def momentum(px: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Invested if the past ~12 months (skipping the last month) return is positive."""
    ret = px.shift(skip) / px.shift(lookback) - 1
    return (ret > 0).astype(float)


def mean_reversion_rsi(px: pd.Series, rsi_len: int = 2, buy_below: float = 10,
                       sell_above: float = 60, trend_window: int = 200) -> pd.Series:
    """Buy sharp dips, but only in an uptrend. Exit once the bounce happened."""
    delta = px.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / rsi_len, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_len, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / down.replace(0, 1e-12))
    uptrend = px > px.rolling(trend_window).mean()
    pos, state = [], 0.0
    for r, up_ok in zip(rsi.values, uptrend.values):
        if state == 0.0 and up_ok and r < buy_below:
            state = 1.0
        elif state == 1.0 and (r > sell_above or not up_ok):
            state = 0.0
        pos.append(state)
    return pd.Series(pos, index=px.index)


STRATEGIES = {
    "buy_and_hold": buy_and_hold,
    "trend_sma200": trend_sma,
    "momentum_12_1": momentum,
    "mean_reversion_rsi2": mean_reversion_rsi,
}
