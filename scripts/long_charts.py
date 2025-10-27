#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S-COIN+ long-term charts generator (1d / 7d / 1m / 1y)
- Uses *_history.csv if available
- Fallback to per-span CSVs
- Maintains UTC→JST shift & trading session filter
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

INDEX_KEY = os.environ.get("INDEX_KEY", "scoin_plus")
OUT_DIR = Path("docs/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_START = os.environ.get("TRADING_START", "09:00")
TRADING_END   = os.environ.get("TRADING_END", "15:30")
INTRADAY_TZ = os.environ.get("INTRADAY_TZ", "UTC")
TZ_OFFSET_HOURS = int(os.environ.get("TZ_OFFSET_HOURS", "9"))

EPS = 5.0
CLAMP_PCT = 30.0

def diag(msg): print(f"[long] {msg}", flush=True)

# ==== theme ====
def apply_dark(fig, ax):
    fig.patch.set_facecolor("#111317")
    ax.set_facecolor("#111317")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="both", colors="#ffffff", labelsize=10)
    ax.yaxis.label.set_color("#ffffff")
    ax.xaxis.label.set_color("#ffffff")
    ax.title.set_color("#ffffff")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.10, color="#ffffff")

# ==== load ====
def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        diag(f"missing: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        return pd.DataFrame()
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    if INTRADAY_TZ.upper() == "UTC":
        df["ts"] = df["ts"] + pd.Timedelta(hours=TZ_OFFSET_HOURS)
    return df

# ==== helpers ====
def session_mask(ts: pd.Series) -> pd.Series:
    sh, sm = map(int, TRADING_START.split(":"))
    eh, em = map(int, TRADING_END.split(":"))
    after_open   = (ts.dt.hour > sh) | ((ts.dt.hour == sh) & (ts.dt.minute >= sm))
    before_close = (ts.dt.hour < eh) | ((ts.dt.hour == eh) & (ts.dt.minute <= em))
    return after_open & before_close

def clamp(p): return max(-CLAMP_PCT, min(CLAMP_PCT, p))
def pct(base, v): return clamp((v - base) / max(abs(base), abs(v), EPS) * 100.0)

# ==== percent series ====
def make_pct(df: pd.DataFrame, span: str) -> pd.DataFrame:
    if df.empty: return df

    if span == "1d":
        day = df["ts"].dt.floor("D").iloc[-1]
        d = df[df["ts"].dt.floor("D") == day].copy()
        d = d.loc[session_mask(d["ts"])].copy()
        if d.empty: return pd.DataFrame()
        base = float(d.iloc[0]["val"])
        d["pct"] = d["val"].apply(lambda v: pct(base, v))
        return d

    days = {"7d": 7, "1m": 30, "1y": 365}.get(span, 7)
    last = df["ts"].max()
    w = df[df["ts"] >= (last - pd.Timedelta(days=days))].copy()
    if w.empty: return w
    base = float(w.iloc[0]["val"])
    w["pct"] = w["val"].apply(lambda v: pct(base, v))
    return w

# ==== plot ====
def plot(dfp: pd.DataFrame, title: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=110)
    apply_dark(fig, ax)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Change (%)", labelpad=10)
    ax.plot(dfp["ts"].values, dfp["pct"].values, linewidth=2.6, color="#ff615a")
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    diag(f"wrote {out_png}")

# ==== main ====
def main():
    spans = ["1d", "7d", "1m", "1y"]
    hist = OUT_DIR / f"{INDEX_KEY}_history.csv"

    # 1️⃣ history 優先
    if hist.exists():
        df_all = load_csv(hist)
        if not df_all.empty:
            for span in spans:
                dfp = make_pct(df_all, span)
                if dfp.empty or "pct" not in dfp:
                    continue
                out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
                plot(dfp, f"{INDEX_KEY.upper()} ({span})", out_png)
            return
        else:
            diag("history.csv empty, fallback to per-span")

    # 2️⃣ fallback: 各span CSV
    for span in spans:
        csv = OUT_DIR / f"{INDEX_KEY}_{span}.csv"
        df = load_csv(csv)
        if df.empty:
            diag(f"skip {span} (no data)")
            continue
        dfp = make_pct(df, span)
        if dfp.empty or "pct" not in dfp:
            diag(f"skip {span} (no pct)")
            continue
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        plot(dfp, f"{INDEX_KEY.upper()} ({span})", out_png)

if __name__ == "__main__":
    diag(f"INDEX_KEY={INDEX_KEY} TZ={INTRADAY_TZ}(+{TZ_OFFSET_HOURS}) SESSION={TRADING_START}-{TRADING_END}")
    main()
