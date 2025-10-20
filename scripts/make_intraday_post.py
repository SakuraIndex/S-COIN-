#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_intraday_post.py

S-COIN+ intraday snapshot generator
-----------------------------------
CSV から以下を生成します：
  1. Intraday snapshot chart (.png)
  2. SNS post text (.txt)
  3. Stats JSON (.json)

特徴:
- JST セッション時間帯の抽出
- 値スケール自動判定（価格 or %）
- 黒背景・枠なしのチャート
- 列名自動検出（記号や大文字小文字を無視）
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt

# === 定数 ===
JST = "Asia/Tokyo"
DATETIME_COL_CANDIDATES = ["Datetime", "datetime", "date", "time", "timestamp"]


# === 引数 ===
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate intraday snapshot & post text")
    p.add_argument("--index-key", required=True, help="index key (e.g. scoin_plus)")
    p.add_argument("--csv", required=True, help="input CSV path")
    p.add_argument("--out-json", required=True, help="output JSON path")
    p.add_argument("--out-text", required=True, help="output text path")
    p.add_argument("--snapshot-png", required=True, help="output PNG path")

    p.add_argument("--session-start", required=True, help="HH:MM JST (e.g. 09:00)")
    p.add_argument("--session-end", required=True, help="HH:MM JST (e.g. 15:30)")
    p.add_argument("--day-anchor", required=True, help="HH:MM JST for labeling (e.g. 09:00)")
    p.add_argument("--basis", required=True, help="basis label (e.g. open@09:00)")
    return p


# === モデル ===
@dataclass
class SessionDef:
    start_h: int
    start_m: int
    end_h: int
    end_m: int
    anchor_h: int
    anchor_m: int

    @staticmethod
    def parse(hhmm_start: str, hhmm_end: str, hhmm_anchor: str) -> "SessionDef":
        def _split(s: str) -> Tuple[int, int]:
            h, m = s.split(":")
            return int(h), int(m)

        sh, sm = _split(hhmm_start)
        eh, em = _split(hhmm_end)
        ah, am = _split(hhmm_anchor)
        return SessionDef(sh, sm, eh, em, ah, am)


# === ユーティリティ ===
def _find_datetime_column(df: pd.DataFrame) -> str:
    for c in DATETIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[0]


def _normalize_key(key: str) -> str:
    return "".join(ch for ch in key if ch.isalnum() or ch == "_").lower()


def _find_value_column(df: pd.DataFrame, index_key: str, dt_col: str) -> str:
    norm_key = _normalize_key(index_key)
    for c in df.columns:
        if c == dt_col:
            continue
        norm_c = _normalize_key(str(c))
        if norm_c == norm_key or norm_c.endswith(norm_key) or norm_key.endswith(norm_c):
            return c
    # fallback: Datetime 以外の 2列目
    non_dt = [c for c in df.columns if c != dt_col]
    if not non_dt:
        raise ValueError("値列が見つかりません")
    return non_dt[0]


def _ensure_jst_index(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    ts = pd.to_datetime(df[dt_col], utc=False, errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(JST)
    else:
        ts = ts.dt.tz_convert(JST)
    out = df.copy()
    out.index = ts
    return out.drop(columns=[dt_col])


def _filter_session(df: pd.DataFrame, sess: SessionDef) -> pd.DataFrame:
    if df.empty:
        return df
    latest_day = df.index.tz_convert(JST)[-1].date()
    start = pd.Timestamp(latest_day, hour=sess.start_h, minute=sess.start_m, tz=JST)
    end = pd.Timestamp(latest_day, hour=sess.end_h, minute=sess.end_m, tz=JST)
    return df.loc[(df.index >= start) & (df.index <= end)].copy()


def _auto_is_percent(series: pd.Series) -> bool:
    mean = float(series.mean())
    diff = float(series.max() - series.min())
    if 50 <= mean <= 200 and diff < 20:
        return True
    return False


def _to_change_percent(series: pd.Series, anchor: float) -> pd.Series:
    if _auto_is_percent(series):
        print("[INFO] Detected % scale — using relative delta")
        return series - series.iloc[0]
    return (series / anchor - 1) * 100.0


# === 出力関数 ===
def save_text(out_path: str, label: str, pct: float, basis: str):
    text = f"▲ {label} 日中スナップショット\n▲ {pct:.2f}%（基準: {basis}）\n#{label} #日本株"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def save_json(out_path: str, index_key: str, label: str, pct: float, sess: SessionDef, now_jst: pd.Timestamp, basis: str):
    obj = {
        "index_key": index_key,
        "label": label,
        "pct_intraday": float(pct),
        "basis": basis,
        "session": {
            "start": f"{sess.start_h:02d}:{sess.start_m:02d}",
            "end": f"{sess.end_h:02d}:{sess.end_m:02d}",
            "anchor": f"{sess.anchor_h:02d}:{sess.anchor_m:02d}"
        },
        "updated_at": now_jst.strftime("%Y-%m-%dT%H:%M:%S%z")
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_chart(out_path: str, series_pct: pd.Series, label: str, now_jst: pd.Timestamp):
    plt.close("all")
    fig = plt.figure(figsize=(11, 6), facecolor="black")
    ax = fig.add_subplot(111, facecolor="black")

    ax.plot(series_pct.index, series_pct.values, linewidth=2, color="#00ffff")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(colors="#cccccc")
    ax.grid(True, alpha=0.15)

    ax.set_title(f"{label} Intraday Snapshot ({now_jst.strftime('%Y/%m/%d %H:%M')})", color="#e6e6e6")
    ax.set_xlabel("Time", color="#cccccc")
    ax.set_ylabel("Change vs Anchor (%)", color="#cccccc")

    fig.tight_layout(pad=1.2)
    fig.savefig(out_path, dpi=150, facecolor="black", edgecolor="none")
    plt.close(fig)


# === メイン ===
def main():
    args = build_parser().parse_args()
    sess = SessionDef.parse(args.session_start, args.session_end, args.day_anchor)

    df = pd.read_csv(args.csv)
    dt_col = _find_datetime_column(df)
    df = _ensure_jst_index(df, dt_col)
    val_col = _find_value_column(df, args.index_key, dt_col)

    df = _filter_session(df, sess)
    if df.empty:
        raise ValueError("セッション内データがありません。")

    label = args.index_key.upper()

    # ✅ tz_localize 修正版（エラー解消）
    now_jst = pd.Timestamp.now(tz="UTC").tz_convert(JST)

    anchor_ts = pd.Timestamp.combine(now_jst.date(), pd.Timestamp(f"{sess.anchor_h:02d}:{sess.anchor_m:02d}").time(), tz=JST)
    anchor_idx = df.index[df.index <= anchor_ts][-1] if (df.index <= anchor_ts).any() else df.index[0]

    series = df[val_col]
    anchor_value = float(series.loc[anchor_idx])
    series_pct = _to_change_percent(series, anchor_value)
    pct_now = float(series_pct.iloc[-1])

    save_text(args.out_text, label, pct_now, args.basis)
    save_json(args.out_json, args.index_key, label, pct_now, sess, now_jst, args.basis)
    save_chart(args.snapshot_png, series_pct, label, now_jst)

    print(f"[DONE] {label}: {pct_now:.2f}% (basis={args.basis})")


if __name__ == "__main__":
    main()
