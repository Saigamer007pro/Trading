"""Reads news headlines with a LOCAL LLM (via Ollama) and turns them into a risk signal.

Example: "NVIDIA CEO steps down unexpectedly"  ->  sentiment -0.8, severity 0.9, action REVIEW/EXIT.

Setup on your PC:  install Ollama (ollama.com), then:  ollama pull llama3.1:8b
Usage:
  python news_ai.py --ticker NVDA                 # live headlines + Ollama
  python news_ai.py --ticker NVDA --mock          # no Ollama, crude keyword scorer (for testing)
"""
import argparse
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"

PROMPT = """You are a cautious equity analyst. Judge how this headline affects the stock {ticker} over the next few days.
Headline: "{headline}"
Reply with JSON only: {{"sentiment": <-1.0 very negative .. +1.0 very positive>,
"severity": <0.0 irrelevant .. 1.0 major company-changing event>, "reason": "<max 15 words>"}}
If the headline is not clearly about {ticker}, give severity 0."""


def fetch_headlines(ticker: str, limit: int = 15) -> list[str]:
    import yfinance as yf
    out = []
    for item in (yf.Ticker(ticker).news or [])[:limit]:
        content = item.get("content", item)
        title = content.get("title")
        if title:
            out.append(title)
    return out


def score_ollama(headline: str, ticker: str, model: str = "llama3.1:8b") -> dict:
    body = json.dumps({"model": model, "stream": False, "format": "json",
                       "options": {"temperature": 0},
                       "messages": [{"role": "user",
                                     "content": PROMPT.format(ticker=ticker, headline=headline)}]}).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = json.loads(r.read())["message"]["content"]
    d = json.loads(text)
    return {"sentiment": max(-1.0, min(1.0, float(d["sentiment"]))),
            "severity": max(0.0, min(1.0, float(d["severity"]))),
            "reason": str(d.get("reason", ""))[:120]}


NEG = ["resign", "steps down", "fraud", "probe", "lawsuit", "recall", "plunge", "crash", "downgrade", "miss", "ban"]
POS = ["beats", "record", "upgrade", "surge", "wins", "approval", "raises guidance"]


def score_mock(headline: str, ticker: str, model: str = "") -> dict:
    h = headline.lower()
    n, p = sum(w in h for w in NEG), sum(w in h for w in POS)
    return {"sentiment": max(-1, min(1, (p - n) * 0.5)), "severity": min(1.0, 0.4 * (n + p)),
            "reason": "keyword match"}


def decide(scores: list[dict]) -> str:
    """Turn many headline scores into one action. Deliberately conservative: news can only
    REDUCE risk or ask for human review, it never opens big positions by itself."""
    if not scores:
        return "HOLD"
    worst = min(scores, key=lambda s: s["sentiment"] * s["severity"])
    impact = worst["sentiment"] * worst["severity"]
    if impact <= -0.5:
        return "EXIT_AND_ASK_HUMAN"
    if impact <= -0.25:
        return "REDUCE_HALF"
    return "HOLD"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    scorer = score_mock if a.mock else score_ollama
    heads = fetch_headlines(a.ticker)
    scores = []
    for h in heads:
        s = scorer(h, a.ticker, a.model)
        scores.append(s)
        print(f"{s['sentiment']:+.2f} sev {s['severity']:.2f} | {h[:90]} | {s['reason']}")
    print("\nACTION:", decide(scores))
