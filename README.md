
# ASX SEPA Scanner v2

A browser-based Streamlit scanner for a mechanical Minervini/SEPA-style workflow.

## What it does

- Downloads daily ASX price history through yfinance.
- Calculates 50/100/150/200-day moving averages.
- Checks Stage-2-style trend structure.
- Measures 6M/12M momentum and relative strength versus STW.
- Measures distance from the 52-week high.
- Uses mechanical proxies for base/VCP contraction:
  - 20/40/80-day range contraction
  - volume quietness
  - ATR contraction
  - higher low
- Calculates a mechanical 20-session pivot.
- Flags breakout + relative-volume conditions.
- Calculates an example risk-based position size.
- Gives explicit statuses:
  - BUY TRIGGER — REVIEW
  - SEPA SETUP
  - WATCH
  - DEVELOPING
  - IGNORE
- Exports the scan to CSV.

## Important limitation

This is NOT a fully validated SEPA implementation.

In particular, it does not automatically retrieve and validate point-in-time:
- EPS growth
- sales growth
- earnings surprises
- institutional sponsorship
- float/share count details
- quality of fundamentals

It also uses mechanical VCP proxies rather than computer vision / discretionary chart interpretation.

Therefore every BUY TRIGGER requires manual fundamental and chart confirmation.

## Deploy

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Open Streamlit Community Cloud.
4. Create app.
5. Select your GitHub repository and `app.py`.
6. Deploy.

Python 3.12 is a sensible current choice on Streamlit Community Cloud.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Custom universe

Upload a CSV with a `ticker`, `symbol`, or `code` column. Both `PME` and `PME.AX` are accepted.

## Data

The default implementation uses yfinance for convenience. For production-grade/live trading use, replace it with a licensed market-data provider. ASX notes that third-party vendors can provide real-time, delayed, or end-of-day ASX data.
