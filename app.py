
import io, math, time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="ASX SEPA Scanner v2", layout="wide")

DEFAULT_TICKERS = """PME
TNE
ALQ
AVH
ANN
CSL
COH
PRU
EVN
BAP
SOP
DXN
360
MAH
SX2
GML
FAU
PAR
PVT
TGN
BCM
COG
JNS
NMG
SHV
ZIP
APX
JIN
STK
ABB
ABG
ADH
AIH
MAF""".split()

st.title("ASX SEPA Scanner v2")
st.caption("Mechanical Minervini/SEPA-style screening tool. It identifies candidates for manual chart review; it is not investment advice and does not place trades.")

with st.sidebar:
    st.header("Scan settings")
    account = st.number_input("Account size (A$)", min_value=0.0, value=50000.0, step=5000.0)
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    max_extension = st.number_input("Max extension above pivot (%)", min_value=1.0, max_value=15.0, value=7.5, step=0.5)
    breakout_vol = st.number_input("Breakout volume / 50D avg", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
    near_high_pct = st.number_input("Near 52W high (%)", min_value=5.0, max_value=40.0, value=25.0, step=1.0)
    min_price = st.number_input("Minimum share price (A$)", min_value=0.01, value=0.50, step=0.10)
    st.divider()
    st.write("Universe")
    uploaded = st.file_uploader("Optional ticker CSV", type=["csv"])
    st.caption("CSV should contain a column named ticker or symbol.")
    refresh = st.button("Run scan", type="primary")

def get_tickers():
    if uploaded:
        df = pd.read_csv(uploaded)
        col = next((c for c in df.columns if c.lower() in ("ticker","symbol","code")), None)
        if col:
            vals = df[col].astype(str).str.upper().str.strip().tolist()
            return sorted(set(x if x.endswith(".AX") else x + ".AX" for x in vals if x and x != "NAN"))
    return [x + ".AX" for x in DEFAULT_TICKERS]

@st.cache_data(ttl=60*60*8, show_spinner=False)
def download_history(ticker):
    try:
        d = yf.download(ticker, period="2y", interval="1d", auto_adjust=False, progress=False, threads=False)
        if d is None or d.empty:
            return None
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        needed = ["Open","High","Low","Close","Volume"]
        d = d[[c for c in needed if c in d.columns]].dropna()
        return d
    except Exception:
        return None

def calc_rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def analyze(ticker, bench):
    d = download_history(ticker)
    if d is None or len(d) < 220:
        return {"Ticker": ticker.replace(".AX",""), "Status":"INSUFFICIENT DATA", "Reason":"<220 trading days"}

    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    for n in (20,50,100,150,200):
        d[f"sma{n}"] = c.rolling(n).mean()

    # Trend / stage
    last = float(c.iloc[-1])
    sma50, sma100, sma150, sma200 = [float(d[f"sma{x}"].iloc[-1]) for x in (50,100,150,200)]
    sma200_20ago = float(d["sma200"].iloc[-21]) if len(d) > 220 else np.nan
    slope200 = (sma200 / sma200_20ago - 1) * 100 if sma200_20ago else np.nan
    trend_gate = last > sma50 > sma150 > sma200 and slope200 > 0

    # Momentum / relative strength
    mom6 = (last / float(c.iloc[-126]) - 1) * 100
    mom12 = (last / float(c.iloc[-252]) - 1) * 100 if len(c) >= 252 else np.nan
    bench_mom6 = np.nan
    if bench is not None and len(bench) >= 126:
        bench_mom6 = (float(bench.iloc[-1]) / float(bench.iloc[-126]) - 1) * 100
    rs6 = mom6 - bench_mom6 if not np.isnan(bench_mom6) else np.nan

    high252 = float(h.tail(252).max())
    low252 = float(l.tail(252).min())
    dist_high = (high252-last)/high252*100
    near_high = dist_high <= near_high_pct

    # Mechanical base / VCP proxies
    r20 = (float(h.tail(20).max()) - float(l.tail(20).min())) / last * 100
    r40 = (float(h.tail(40).max()) - float(l.tail(40).min())) / last * 100
    r80 = (float(h.tail(80).max()) - float(l.tail(80).min())) / last * 100
    range_contract = r20 < r40 < r80
    vol20 = float(v.tail(20).mean())
    vol50 = float(v.tail(50).mean())
    vol_quiet = vol20 < vol50 * 0.85
    atr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr20, atr50 = float(atr.tail(20).mean()), float(atr.tail(50).mean())
    atr_contract = atr20 < atr50 * 0.90
    higher_low = float(l.tail(20).min()) >= float(l.iloc[-40:-20].min())

    # Pivot: highest high of prior 20 sessions, excluding today
    pivot = float(h.iloc[-21:-1].max())
    today_vol_ratio = float(v.iloc[-1] / vol50) if vol50 else np.nan
    breakout = last > pivot and today_vol_ratio >= breakout_vol
    extension = (last/pivot - 1)*100 if pivot else np.nan

    # Earnings/fundamentals are intentionally not inferred from price data.
    # We expose a hard "fundamental verification" gate so a user cannot mistake this for a complete SEPA model.
    fundamental_verified = False

    trend_points = sum([last>sma50, sma50>sma150, sma150>sma200, slope200>0])
    setup_points = sum([range_contract, vol_quiet, atr_contract, higher_low])
    momentum_points = sum([mom6>0, (not np.isnan(rs6) and rs6>0), near_high])

    if last < 0.5:
        status, reason = "IGNORE", "Below minimum price"
    elif trend_gate and breakout and extension <= max_extension and momentum_points >= 2:
        status, reason = "BUY TRIGGER — REVIEW", "Trend + pivot breakout + volume; verify fundamentals and chart"
    elif trend_gate and momentum_points >= 2 and setup_points >= 2:
        status, reason = "SEPA SETUP", "Stage-2 trend and tightening setup; wait for clean pivot breakout"
    elif trend_points >= 3 and momentum_points >= 2:
        status, reason = "WATCH", "Strong trend/momentum but setup is incomplete"
    elif momentum_points >= 2:
        status, reason = "DEVELOPING", "Momentum exists, but Stage-2/structure gate is incomplete"
    else:
        status, reason = "IGNORE", "Insufficient trend/momentum"

    risk_amount = account * risk_pct / 100
    stop = min(float(l.tail(20).min()), last * 0.93)
    per_share_risk = max(last-stop, 0.01)
    shares = math.floor(risk_amount/per_share_risk) if account else 0

    return {
        "Ticker": ticker.replace(".AX",""),
        "Status": status,
        "Price": round(last,3),
        "50DMA": round(sma50,3),
        "150DMA": round(sma150,3),
        "200DMA": round(sma200,3),
        "200DMA slope %": round(slope200,2),
        "6M %": round(mom6,1),
        "12M %": round(mom12,1) if not np.isnan(mom12) else np.nan,
        "RS vs bench 6M %": round(rs6,1) if not np.isnan(rs6) else np.nan,
        "From 52W high %": round(dist_high,1),
        "20D range %": round(r20,1),
        "40D range %": round(r40,1),
        "80D range %": round(r80,1),
        "Volume quiet": "Yes" if vol_quiet else "No",
        "ATR contract": "Yes" if atr_contract else "No",
        "Higher low": "Yes" if higher_low else "No",
        "Pivot": round(pivot,3),
        "Vol / 50D": round(today_vol_ratio,2) if not np.isnan(today_vol_ratio) else np.nan,
        "Extension %": round(extension,1) if not np.isnan(extension) else np.nan,
        "Stop est.": round(stop,3),
        "Risk A$": round(risk_amount,2),
        "Shares @ risk": shares,
        "Fundamentals": "VERIFY MANUALLY",
        "Reason": reason
    }

def main():
    tickers = get_tickers()
    st.write(f"Scanning **{len(tickers)} tickers** using daily data.")
    with st.spinner("Downloading price history and calculating SEPA-style conditions..."):
        bench_df = download_history("STW.AX")
        bench = bench_df["Close"] if bench_df is not None else None
        rows = []
        progress = st.progress(0)
        for i,t in enumerate(tickers):
            rows.append(analyze(t, bench))
            progress.progress((i+1)/len(tickers))
        progress.empty()

    out = pd.DataFrame(rows)
    order = {"BUY TRIGGER — REVIEW":0, "SEPA SETUP":1, "WATCH":2, "DEVELOPING":3, "IGNORE":4, "INSUFFICIENT DATA":5}
    out["rank"] = out["Status"].map(order).fillna(9)
    out = out.sort_values(["rank","6M %"], ascending=[True,False]).drop(columns="rank")

    st.subheader("Scan results")
    st.dataframe(out, use_container_width=True, hide_index=True)

    candidates = out[out["Status"].isin(["BUY TRIGGER — REVIEW","SEPA SETUP","WATCH"])]
    st.subheader("Priority review list")
    if candidates.empty:
        st.info("No high-priority candidates passed the mechanical gates.")
    else:
        st.dataframe(candidates[["Ticker","Status","Price","50DMA","150DMA","200DMA","200DMA slope %","6M %","RS vs bench 6M %","From 52W high %","Pivot","Vol / 50D","Extension %","Fundamentals","Reason"]], use_container_width=True, hide_index=True)

    st.subheader("How to use a BUY TRIGGER")
    st.markdown("""
**Do not treat the green label as an automatic buy.** Before acting, manually verify:
1. The company has strong and improving earnings/revenue characteristics.
2. The chart is a genuine constructive base/VCP rather than a volatile spike.
3. The pivot is obvious on the chart.
4. The breakout is decisive and not materially extended.
5. The stop is placed at a logical technical level.
6. Position size keeps the loss at or below your chosen account-risk percentage.

The current fundamental field deliberately says **VERIFY MANUALLY** because this version does not pretend that free price-history data is a reliable substitute for point-in-time fundamental data.
""")

    csv = out.to_csv(index=False).encode()
    st.download_button("Download scan CSV", csv, "asx_sepa_scan.csv", "text/csv")

if __name__ == "__main__":
    main()
