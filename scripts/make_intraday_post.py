#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_intraday_post.py

Intraday CSV を読み取り、投稿テキスト (.txt)・統計 (.json)・スナップショット画像 (.png) を生成。
日本株（JST 09:00–15:30）を基本としつつ、24h銘柄にも対応。
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

JST = "Asia/Tokyo"

# ============================================================
# Utility functions
# ============================================================

def _find_ts_and_value_columns(df: pd.DataFrame,
                               index_key: str | None = None) -> tuple[str, str]:
    """
    時刻列と値列を推定。
    """
    lowered = {c.lower(): c for c in df.columns}

    # 時刻列候補
    for cand in ["ts", "timestamp", "time", "datetime", "date", "time_jst"]:
        if cand in lowered:
            ts_col = lowered[cand]
            break
    else:
        raise ValueError("時刻列(ts/timestamp/time/datetime/date/time_jst)が見つかりません")

    # 値列推定
    value_col = None
    if index_key:
        for c in df.columns:
            if c.lower() == index_key.lower() or c.lower().replace("+", "").replace("-", "") == index_key.lower():
                value_col = c
                break

    if value_col is None and len(df.columns) == 2:
        value_col = [c for c in df.columns if c != ts_col][0]

    if value_col is None:
        for c in df.columns:
            if c == ts_col:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                value_col = c
                break

    if value_col is None:
        raise ValueError("値列が特定できません (index_key または数値列を確認)")

    return ts_col, value_col


def load_intraday(csv_path: str, index_key: str | None = None) -> pd.DataFrame:
    """CSV 読み込みと時刻変換 (JST)"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    ts_col, value_col = _find_ts_and_value_columns(df, index_key=index_key)

    ts = pd.to_datetime(df[ts_col], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(JST, nonexistent="shift_forward", ambiguous="NaT")
    else:
        ts = ts.dt.tz_convert(JST)

    out = pd.DataFrame({
        "ts": ts,
        "value": pd.to_numeric(df[value_col], errors="coerce")
    }).dropna(subset=["ts"])

    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return out


def filter_session(df: pd.DataFrame, session: str | None) -> pd.DataFrame:
    """指定セッション（例: '09:00-15:30'）で絞り込み"""
    if not session or session.lower() in ["24h", "all"]:
        return df

    if "-" not in session:
        raise ValueError(f"セッション指定の形式が不正: {session}")

    start, end = session.split("-")
    start_h, start_m = [int(x) for x in start.split(":")]
    end_h, end_m = [int(x) for x in end.split(":")]

    day = df["ts"].iloc[-1].tz_convert(JST)
    st = pd.Timestamp(day.year, day.month, day.day, start_h, start_m, tz=JST)
    en = pd.Timestamp(day.year, day.month, day.day, end_h, end_m, tz=JST)
    return df[(df["ts"] >= st) & (df["ts"] <= en)].copy()


# ============================================================
# Core calculations
# ============================================================

def calc_change_stats(df: pd.DataFrame) -> dict:
    """セッション中の値変化・統計を算出"""
    if df.empty:
        return {"change_pct": np.nan, "open": np.nan, "close": np.nan}

    open_val = df["value"].iloc[0]
    close_val = df["value"].iloc[-1]
    change_pct = (close_val / open_val - 1.0) * 100.0 if open_val != 0 else np.nan

    return {
        "open": round(open_val, 4),
        "close": round(close_val, 4),
        "change_pct": round(change_pct, 2),
    }


def plot_snapshot(df: pd.DataFrame, label: str, out_png: str):
    """Intraday Snapshot PNG生成"""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)

    if df.empty:
        ax.text(0.5, 0.5, "No intraday data", ha="center", va="center", fontsize=14)
    else:
        pct = (df["value"] / df["value"].iloc[0] - 1.0) * 100.0
        ax.plot(df["ts"].dt.tz_localize(None), pct, color="cyan", linewidth=2, label=label)

    now_str = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(JST).strftime("%Y/%m/%d %H:%M")
    ax.set_title(f"{label} Intraday Snapshot ({now_str})", fontsize=14, pad=10)
    ax.set_xlabel("Time (JST)")
    ax.set_ylabel("Change vs Open (%)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Output generation
# ============================================================

def generate_post_text(index_key: str, stats: dict, out_text: str, label: str):
    now_jst = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(JST)
    time_str = now_jst.strftime("%Y/%m/%d %H:%M")

    trend_symbol = "▲" if stats["change_pct"] > 0 else "▼" if stats["change_pct"] < 0 else "■"
    change_str = f"{stats['change_pct']:+.2f}%" if not np.isnan(stats["change_pct"]) else "N/A"

    lines = [
        f"{trend_symbol} {label} 日中取引 ({time_str})",
        f"{change_str}（前日終値比）",
        f"※ 構成銘柄の等ウェイト平均",
        f"#{label.replace(' ', '')} #日本株"
    ]

    with open(out_text, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser(description="Generate intraday post and snapshot")
    p.add_argument("--index-key", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-text", required=True)
    p.add_argument("--snapshot-png", default=None)
    p.add_argument("--session", default="09:00-15:30")
    p.add_argument("--label", default=None)
    args = p.parse_args()

    label = args.label or args.index_key.upper()

    df = load_intraday(args.csv, index_key=args.index_key)
    df = filter_session(df, args.session)
    stats = calc_change_stats(df)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    generate_post_text(args.index_key, stats, args.out_text, label)

    if args.snapshot_png:
        plot_snapshot(df, label, args.snapshot_png)

    print(f"✅ Done: {args.index_key} {stats['change_pct']}%")


if __name__ == "__main__":
    main()
