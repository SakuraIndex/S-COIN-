#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

INDEX_KEY = os.environ.get("INDEX_KEY", "scoin_plus")
OUT_DIR = Path("docs/outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_START = os.environ.get("TRADING_START", "09:00")
TRADING_END   = os.environ.get("TRADING_END",   "15:30")
INTRADAY_TZ = os.environ.get("INTRADAY_TZ", "UTC")
TZ_OFFSET_HOURS = int(os.environ.get("TZ_OFFSET_HOURS", "9"))

EPS = 5.0
CLAMP_PCT = 30.0

def apply_dark(fig, ax):
    fig.patch.set_facecolor("#111317")
    ax.set_facecolor("#111317")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(axis="both", colors="#ffffff", labelsize=10)
    ax.yaxis.label.set_color("#ffffff"); ax.xaxis.label.set_color("#ffffff"); ax.title.set_color("#ffffff")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.10, color="#ffffff")

def load_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col:"ts", val_col:"val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts","val"]).sort_values("ts").reset_index(drop=True)
    if INTRADAY_TZ.upper() == "UTC":
        df["ts"] = df["ts"] + pd.Timedelta(hours=TZ_OFFSET_HOURS)
    return df

def session_mask(series: pd.Series) -> pd.Series:
    sh, sm = map(int, TRADING_START.split(":"))
    eh, em = map(int, TRADING_END.split(":"))
    after_open  = (series.dt.hour > sh) | ((series.dt.hour == sh) & (series.dt.minute >= sm))
    before_close= (series.dt.hour < eh) | ((series.dt.hour == eh) & (series.dt.minute <= em))
    return after_open & before_close

def clamp_pct(p):
    if p > CLAMP_PCT: return CLAMP_PCT
    if p < -CLAMP_PCT: return -CLAMP_PCT
    return p

def make_pct(df: pd.DataFrame, span: str) -> pd.DataFrame:
    if df.empty: return df
    if span == "1d":
        day = df["ts"].dt.floor("D").iloc[-1]
        d = df[df["ts"].dt.floor("D")==day].copy()
        d = d.loc[session_mask(d["ts"])]
        if d.empty: return d
        base = float(d.iloc[0]["val"])
        d["pct"] = d["val"].apply(lambda v: clamp_pct((v - base)/max(abs(base),abs(v),EPS)*100.0))
        return d
    else:
        last = df["ts"].max()
        dur = {"7d":7,"1m":30,"1y":365}.get(span,7)
        w = df[df["ts"] >= (last - pd.Timedelta(days=dur))].copy()
        if w.empty: return w
        base = float(w.iloc[0]["val"])
        w["pct"] = w["val"].apply(lambda v: clamp_pct((v - base)/max(abs(base),abs(v),EPS)*100.0))
        return w

def plot(dfp: pd.DataFrame, title: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(16,8), dpi=110)
    apply_dark(fig, ax)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10); ax.set_ylabel("Change (%)", labelpad=10)
    ax.plot(dfp["ts"].values, dfp["pct"].values, linewidth=2.6, color="#ff615a")
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major); ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=50)); ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    fig.tight_layout(); fig.savefig(out_png, facecolor=fig.get_facecolor()); plt.close(fig)

def main():
    for span in ["1d","7d","1m","1y"]:
        csv = OUT_DIR / f"{INDEX_KEY}_{span}.csv"
        if not csv.exists(): continue
        df = load_csv(csv); dfp = make_pct(df, span)
        if dfp is None or dfp.empty or "pct" not in dfp: continue
        plot(dfp, f"{INDEX_KEY.upper()} ({span})", OUT_DIR / f"{INDEX_KEY}_{span}.png")

if __name__ == "__main__":
    main()
