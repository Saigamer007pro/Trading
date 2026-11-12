# Trading bot (Phase 1: research only, no real money)

## One-time setup on Windows
1. Install Python 3.12+ from python.org (tick "Add Python to PATH").
2. Install Ollama from ollama.com, then in a terminal: `ollama pull llama3.1:8b`
3. In this folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Run
- Backtest strategies on real data: `python run_backtest.py --ticker SPY`
- Read news with the local AI: `python news_ai.py --ticker NVDA`

## Files
- data.py: downloads prices (cached in ./data)
- strategies.py: simple baseline strategies
- backtest.py: honest backtester (no lookahead, includes trading costs)
- run_backtest.py: compares strategies in-sample vs out-of-sample
- news_ai.py: local LLM (Ollama) reads headlines; can only reduce risk or ask you
