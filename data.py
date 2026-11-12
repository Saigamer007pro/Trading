"""Price data: download real data (cached to disk) or generate synthetic data for testing."""
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path(__file__).parent / "data"


def load_prices(ticker: str, start: str = "2010-01-01") -> pd.Series:
    """Daily adjusted close prices for a US ticker. Cached in ./data after first download."""
    import yfinance as yf
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{ticker}.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0, parse_dates=True)["close"]
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}. Check internet / ticker.")
    close = df["Close"].squeeze().rename("close").dropna()
    close.to_frame().to_csv(f)
    return close


def synthetic_prices(n_days: int = 3000, seed: int = 0) -> pd.Series:
    """Fake price series with slow drift, volatility clusters and regime changes. For TESTING code only."""
    rng = np.random.default_rng(seed)
    vol = np.empty(n_days)
    v = 0.01
    for i in range(n_days):
        v = 0.98 * v + 0.02 * 0.01 + 0.0001 * abs(rng.standard_normal())
        vol[i] = v
    drift = 0.0004 + 0.0006 * np.sin(np.arange(n_days) / 250)
    rets = drift + vol * rng.standard_normal(n_days)
    idx = pd.bdate_range("2010-01-01", periods=n_days)
    return pd.Series(100 * np.exp(np.cumsum(rets)), index=idx, name="close")
