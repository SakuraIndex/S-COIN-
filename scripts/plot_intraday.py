#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_intraday.py

intraday CSV を読み、セッションで絞り込み、
- 既存の変化率列(change_pct/pct/...) があればそれを描画
- なければ値列からセッション先頭を100%基準にした変化率を算出して描画
PNG スナップショットを出力します。

Astra4 / R-BANK9 / S-COIN+ 共通で利用可（日本株はデフォで 09:00–15:30 JST）。
"""

from __future__ import annotations
import argparse
from typing import Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

JST = "Asia/Tokyo"

# ====== utils ======

def _to_jst(ts: pd.Series) -> pd.Series:
    s = ts.copy()
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    return s.dt.tz_convert(JST)

def _detect_ts_column(df: pd.DataFrame) -> str:
    for c in ["ts", "timestamp", "time", "datetime"]:
        if c in df.columns:
            return c
    raise ValueError("時刻列(ts/timestamp/time/datetime)が見つかりません")

def _detect_pct_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["change_pct", "pct", "chg_pct", "pct_chg"]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def _detect_value_column(df: pd.DataFrame) -> str:
    for c in ["level", "value", "close", "price", "index", "idx"]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    # フォールバック: 最初の数値列
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("数値列が見つかりません")

def _session_range(day_ts: pd.Timestamp, start_hm: str, end_hm: str):
    sh, sm = [int(x) for x in start_hm.split(":")]
    eh, em = [int(x) for x in end_hm.split(":")]
    st = pd.Timestamp(day_ts.year, day_ts.month, day_ts.day, sh, sm, tz=JST)
    en = pd.Timestamp(day_ts.year, day_ts.month, day_ts.day, eh, em, tz=JST)
    return st, en

# ====== core ======

def load_intraday(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    ts_col = _detect_ts_column(df)
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).copy()
    df["ts"] = _to_jst(df[ts_col])
    return df.sort_values("ts").reset_index(drop=True)

def filter_session(df: pd.DataFrame, mode: str, start: str, end: str) -> pd.DataFrame:
    if df.empty:
        return df
    if mode == "24h":
        return df
    day = df["ts"].iloc[-1].tz_convert(JST)
    st, en = _session_range(day, start, end)
    return df[(df["ts"] >= st) & (df["ts"] <= en)].copy()

def to_plot_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    """
    戻り値: (ts_series, y_series_percent, ylabel)
    """
    pct_col = _detect_pct_column(df)
    if pct_col is not None:
        return df["ts"], df[pct_col].astype(float), "Change vs Prev Close (%)"
    # 値列からセッション先頭を1.0として変化率[%]を計算
    val_col = _detect_value_column(df)
    vals = df[val_col].astype(float)
    if len(vals) == 0 or not np.isfinite(vals.iloc[0]) or vals.iloc[0] == 0:
        y = pd.Series(np.nan, index=df.index)
    else:
        y = (vals / vals.iloc[0] - 1.0) * 100.0
    return df["ts"], y, "Change (%)"

# ====== plotting ======

def plot_snapshot(
    csv_path: str,
    out_png: str,
    label: str,
    session_mode: str = "stock",
    session_start: str = "09:00",
    session_end: str = "15:30",
) -> None:
    df = load_intraday(csv_path)
    df = filter_session(df, session_mode, session_start, session_end)
    if df.empty:
        # 空でも「枠だけ」出しておく
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(12, 6), dpi=140)
        ax = fig.add_subplot(111)
        ax.set_title(f"{label} Intraday Snapshot (NO DATA)", fontsize=14, pad=10)
        ax.set_xlabel("Time")
        ax.set_ylabel("Change (%)")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        return

    ts, y, ylabel = to_plot_series(df)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12, 6), dpi=140)
    ax = fig.add_subplot(111)

    # 線
    ax.plot(ts.dt.tz_convert(JST).dt.tz_localize(None), y, linewidth=2.0, label=label)

    # 体裁
    now_str = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(JST).strftime("%Y/%m/%d %H:%M")
    ax.set_title(f"{label} Intraday Snapshot ({now_str})", fontsize=14, pad=10)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.1)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ====== CLI ======

def parse_args():
    p = argparse.ArgumentParser(description="Plot intraday snapshot PNG")
    p.add_argument("--csv", required=True, help="intraday CSV path")
    p.add_argument("--out", required=True, help="output PNG")
    p.add_argument("--label", required=True, help="legend/series label")
    p.add_argument("--session", default="stock", choices=["stock", "24h"])
    p.add_argument("--session-start", default="09:00")
    p.add_argument("--session-end", default="15:30")
    return p.parse_args()

def main():
    a = parse_args()
    plot_snapshot(
        csv_path=a.csv,
        out_png=a.out,
        label=a.label,
        session_mode=a.session,
        session_start=a.session_start,
        session_end=a.session_end,
    )

if __name__ == "__main__":
    main()
