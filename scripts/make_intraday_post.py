#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_intraday_post.py (修正版)
- 背景: 黒ベースに戻した
- 騰落率: 既に%値を持つデータに対して二重変換しないように修正
"""

import argparse
import json
import math
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

JST = "Asia/Tokyo"


def norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def resolve_column(index_key: str, columns) -> str:
    alias = {
        "scoin_plus": "S-COIN+",
        "rbank9": "R-BANK9",
        "ain10": "AIN-10",
    }
    if index_key in alias and alias[index_key] in columns:
        return alias[index_key]

    target = norm_token(index_key)
    for c in columns:
        if norm_token(c) == target:
            return c
    raise ValueError(f"CSVに '{index_key}' に対応する列が見つかりません。候補: {list(columns)}")


def parse_hhmm(s: str) -> Tuple[int, int]:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s.strip())
    if not m:
        raise ValueError(f"時刻はHH:MM形式で指定してください: {s!r}")
    hh = int(m.group(1))
    mm = int(m.group(2))
    return hh, mm


def utcnow_jst() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(JST)


@dataclass
class Args:
    index_key: str
    csv: str
    out_text: str
    out_json: Optional[str]
    snapshot_png: Optional[str]
    label: Optional[str]
    session_start: str
    session_end: str
    day_anchor: str
    basis: str


# === CSV読み込み ===
def load_intraday(csv_path: str, index_key: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Datetime" not in df.columns:
        raise ValueError("CSVに 'Datetime' 列が必要です。")

    col = resolve_column(index_key, df.columns)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime"]).copy()
    df["Datetime"] = df["Datetime"].dt.tz_convert(JST)
    df = df[["Datetime", col]].rename(columns={col: "value"}).sort_values("Datetime")
    return df


def filter_session(df: pd.DataFrame, session_start: str, session_end: str) -> pd.DataFrame:
    if df.empty:
        return df
    h1, m1 = parse_hhmm(session_start)
    h2, m2 = parse_hhmm(session_end)
    d = df["Datetime"].iloc[-1].date()
    start = pd.Timestamp(d.year, d.month, d.day, h1, m1, tz=JST)
    end = pd.Timestamp(d.year, d.month, d.day, h2, m2, tz=JST)
    return df[(df["Datetime"] >= start) & (df["Datetime"] <= end)].copy()


def find_anchor_value(df: pd.DataFrame, anchor_hhmm: str) -> Tuple[pd.Timestamp, float]:
    hh, mm = parse_hhmm(anchor_hhmm)
    d = df["Datetime"].iloc[-1].date()
    anchor_ts = pd.Timestamp(d.year, d.month, d.day, hh, mm, tz=JST)
    df_after = df[df["Datetime"] >= anchor_ts]
    row = df_after.iloc[0] if not df_after.empty else df.iloc[-1]
    return row["Datetime"], float(row["value"])


def pct_change(cur: float, base: float) -> float:
    if abs(base) < 1e-8 or not np.isfinite(base) or not np.isfinite(cur):
        return float("nan")
    return (cur / base - 1.0) * 100.0


# === 出力 ===
def make_post_text(label: str, pct_intraday: float, now_ts: pd.Timestamp, basis_label: str) -> str:
    arrow = "▲" if pct_intraday >= 0 else "▼"
    pct_str = f"{pct_intraday:+.2f}%"
    jst_str = now_ts.strftime("%Y/%m/%d %H:%M")
    return f"{arrow} {label} 日中スナップショット ({jst_str})\n{pct_str}（基準: {basis_label}）\n#{label.replace(' ', '_')} #日本株\n"


def save_snapshot_png(df: pd.DataFrame, png_path: str, label: str, anchor_label: str):
    if df.empty:
        raise ValueError("データが空です。")

    # FIX: 値が既に%表現（±数％）の場合、そのまま使う
    y = df["value"].to_numpy()
    if np.nanmax(np.abs(y)) > 20:  # 値がすでに変化率でない（指数など）場合のみ再計算
        _, base_v = find_anchor_value(df, anchor_label.split("@")[-1])
        y = (df["value"] / base_v - 1.0) * 100.0

    plt.style.use("dark_background")  # FIX: 黒ベース背景
    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    ax.plot(df["Datetime"], y, color="cyan", linewidth=2)
    ax.set_title(f"{label} Intraday Snapshot ({df['Datetime'].iloc[-1].strftime('%Y/%m/%d %H:%M')})")
    ax.set_ylabel("Change vs Anchor (%)")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close()


# === メイン ===
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-key", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-text", required=True)
    p.add_argument("--out-json", default=None)
    p.add_argument("--snapshot-png", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--session-start", required=True)
    p.add_argument("--session-end", required=True)
    p.add_argument("--day-anchor", required=True)
    p.add_argument("--basis", required=True)
    a = p.parse_args()

    args = Args(**vars(a))
    label = args.label or args.index_key.upper()

    df_all = load_intraday(args.csv, args.index_key)
    df = filter_session(df_all, args.session_start, args.session_end)
    if df.empty:
        raise ValueError("セッション内にデータがありません。")

    anchor_ts, anchor_val = find_anchor_value(df, args.day_anchor)
    now_val = float(df.iloc[-1]["value"])
    now_ts = df.iloc[-1]["Datetime"]

    # FIX: 値がすでに%の場合はそのまま
    if np.nanmax(np.abs(df["value"])) > 20:
        pct_intraday = now_val - anchor_val
    else:
        pct_intraday = pct_change(now_val, anchor_val)

    text = make_post_text(label, pct_intraday, now_ts, args.basis)
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write(text)

    if args.snapshot_png:
        save_snapshot_png(df, args.snapshot_png, label, args.basis)

    if args.out_json:
        out = {
            "index_key": args.index_key,
            "label": label,
            "pct_intraday": pct_intraday,
            "basis": args.basis,
            "session": {
                "start": args.session_start,
                "end": args.session_end,
                "anchor": args.day_anchor,
            },
            "updated_at": now_ts.isoformat(),
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
