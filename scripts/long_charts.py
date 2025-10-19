#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

# ==== 環境変数 ====
INDEX_KEY = os.environ.get("INDEX_KEY", "scoin_plus")
OUT_DIR = Path("docs/outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# 取引時間（JST 想定）
TRADING_START = os.environ.get("TRADING_START", "09:00")
TRADING_END   = os.environ.get("TRADING_END",   "15:30")

# CSV の時刻タイムゾーン（UTC の場合は +9h してから 09:00–15:30 を切る）
INTRADAY_TZ = os.environ.get("INTRADAY_TZ", "UTC")   # "UTC" | "JST"
TZ_OFFSET_HOURS = int(os.environ.get("TZ_OFFSET_HOURS", "9"))

# 安全パラメータ
EPS = 5.0            # 極小分母回避
CLAMP_PCT = 30.0     # 表示用クランプ（±30%）

def diag(msg: str):
    print(f"[long_charts] {msg}", flush=True)

# ==== 描画テーマ（白文字） ====
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

# ==== 入出力 ====
def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        diag(f"CSV not found: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        diag(f"CSV has <2 columns: {csv_path}")
        return pd.DataFrame()
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df = df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)
    # UTC → JST シフト
    if INTRADAY_TZ.upper() == "UTC":
        df["ts"] = df["ts"] + pd.Timedelta(hours=TZ_OFFSET_HOURS)
    return df

def session_mask(ts: pd.Series) -> pd.Series:
    sh, sm = map(int, TRADING_START.split(":"))
    eh, em = map(int, TRADING_END.split(":"))
    after_open   = (ts.dt.hour > sh) | ((ts.dt.hour == sh) & (ts.dt.minute >= sm))
    before_close = (ts.dt.hour < eh) | ((ts.dt.hour == eh) & (ts.dt.minute <= em))
    return after_open & before_close

def clamp_pct(p):
    if p > CLAMP_PCT: return CLAMP_PCT
    if p < -CLAMP_PCT: return -CLAMP_PCT
    return p

# ==== ％系列作成 ====
def make_pct(df: pd.DataFrame, span: str) -> pd.DataFrame:
    if df.empty:
        return df

    if span == "1d":
        # 当日のみ + セッション内
        day = df["ts"].dt.floor("D").iloc[-1]
        d = df[df["ts"].dt.floor("D") == day].copy()
        d_sess = d.loc[session_mask(d["ts"])].copy()
        diag(f"1d rows: total={len(d)}, session={len(d_sess)}")
        if d_sess.empty:
            return pd.DataFrame()
        base = float(d_sess.iloc[0]["val"])
        denom = lambda v: max(abs(base), abs(v), EPS)
        d_sess["pct"] = d_sess["val"].apply(lambda v: clamp_pct((v - base) / denom(v) * 100.0))
        return d_sess

    # 7d/1m/1y は期間先頭値を基準に％
    last = df["ts"].max()
    days = {"7d": 7, "1m": 30, "1y": 365}.get(span, 7)
    w = df[df["ts"] >= (last - pd.Timedelta(days=days))].copy()
    diag(f"{span} rows: {len(w)}")
    if w.empty: return w
    base = float(w.iloc[0]["val"])
    denom = lambda v: max(abs(base), abs(v), EPS)
    w["pct"] = w["val"].apply(lambda v: clamp_pct((v - base) / denom(v) * 100.0))
    return w

# ==== 描画 ====
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
    diag(f"WROTE: {out_png}")

def main():
    spans = ["1d", "7d", "1m", "1y"]
    for span in spans:
        # 1d は通常の *_1d.csv を使用（タイムシフト & セッションフィルタ対応）
        csv = OUT_DIR / f"{INDEX_KEY}_{span}.csv"
        df = load_csv(csv)
        if df.empty:
            diag(f"Skip {span}: empty or missing CSV")
            continue
        dfp = make_pct(df, span)
        if dfp is None or dfp.empty or "pct" not in dfp:
            diag(f"Skip {span}: no pct series (after session/period filter)")
            continue
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        plot(dfp, f"{INDEX_KEY.upper()} ({span})", out_png)

if __name__ == "__main__":
    diag(f"INDEX_KEY={INDEX_KEY}  TZ={INTRADAY_TZ}(+{TZ_OFFSET_HOURS})  "
         f"SESSION={TRADING_START}-{TRADING_END}")
    main()
