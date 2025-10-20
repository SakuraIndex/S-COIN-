#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intraday snapshot & post generator (JST session aware)

- CSV の Datetime を UTC/naive/tz-aware いずれでも受け取り、JST に正規化
- 対象列名は柔軟にマッチ（例: "S-COIN+", "scoin_plus" など）
- 当日の判定は「CSV の最新時刻(JST)の '日'」に合わせてセッション窓を構築
- セッション境界は包含 (>=, <=) でフィルタして取りこぼしを回避
- セッションが空でもワーニングで継続（PNG/JSON/TXT は必ず作成）
- グラフは黒背景・白枠なし（外周スパイン非表示）

依存: pandas, numpy, matplotlib, pytz
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt

JST = pytz.timezone("Asia/Tokyo")


# ---------- 引数 ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make intraday snapshot and post text (JST session aware)")
    p.add_argument("--index-key", required=True, help="Logical index key (e.g., scoin_plus)")
    p.add_argument("--csv", required=True, help="Input intraday CSV path")
    p.add_argument("--out-json", required=True, help="Output stats JSON path")
    p.add_argument("--out-text", required=True, help="Output post text path")
    p.add_argument("--snapshot-png", required=True, help="Output snapshot PNG path")
    p.add_argument("--session-start", required=True, help="JST session start (HH:MM)")
    p.add_argument("--session-end", required=True, help="JST session end (HH:MM)")
    p.add_argument("--day-anchor", required=True, help="Day anchor for labeling (HH:MM JST)")
    p.add_argument("--basis", required=True, help='Return basis label, e.g., "open@09:00"')
    return p.parse_args()


# ---------- ユーティリティ ----------
def _to_ts_jst(x) -> pd.Timestamp:
    """CSV の Datetime を JST に正規化"""
    ts = pd.to_datetime(x, utc=None, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:  # naive → UTC とみなす
        ts = ts.tz_localize("UTC").tz_convert(JST)
    else:  # tz-aware
        ts = ts.tz_convert(JST)
    return ts


def _norm_name(s: str) -> str:
    """列名や index_key をゆるく正規化"""
    return str(s).strip().lower().replace("-", "_").replace("+", "plus").replace(" ", "_")


def load_intraday(csv_path: str, index_key: str) -> pd.DataFrame:
    """CSV を読み、時刻を JST に、値列を 'value' に正規化して返す"""
    df = pd.read_csv(csv_path)

    # 時刻列を推定（Datetime/Date/Time/Timestamp など）
    time_col = None
    for c in df.columns:
        if _norm_name(c) in ("datetime", "date", "time", "timestamp"):
            time_col = c
            break
    if time_col is None:
        time_col = df.columns[0]  # 先頭を時刻とみなすフォールバック

    # 値列を推定（index_key を正規化してマッチ）
    target = _norm_name(index_key)
    value_col = None
    for c in df.columns:
        if _norm_name(c) == target:
            value_col = c
            break
    if value_col is None:
        raise ValueError(f"CSV から '{index_key}' に対応する列を特定できません。候補: {', '.join(df.columns)}")

    # JST 時刻化
    df["ts"] = df[time_col].apply(_to_ts_jst)
    df = df.rename(columns={value_col: "value"})[["ts", "value"]]
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


def _today_from_data(df_jst: pd.DataFrame) -> pd.Timestamp:
    """データの最新行の '日'(JST) を返す。データが無い場合は今日(JST)"""
    latest = df_jst["ts"].max()
    if pd.isna(latest):
        return pd.Timestamp.now(tz=JST).normalize()
    return latest.normalize()


def _at_day(hhmm: str, day: pd.Timestamp) -> pd.Timestamp:
    hh, mm = map(int, hhmm.split(":"))
    return day + pd.Timedelta(hours=hh, minutes=mm)


def build_session_window(
    df_jst: pd.DataFrame, day_anchor_hhmm: str, sess_start_hhmm: str, sess_end_hhmm: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    その日の判定は「CSV 最新時刻(JST) の '日'」。
    end <= start の場合は 12h 先送り（念のため）
    """
    base_day = _today_from_data(df_jst)
    start = _at_day(sess_start_hhmm, base_day)
    end = _at_day(sess_end_hhmm, base_day)
    if end <= start:
        end = end + pd.Timedelta(hours=12)
    return start, end


def filter_session(
    df_jst: pd.DataFrame, sess_start_hhmm: str, sess_end_hhmm: str, day_anchor_hhmm: str
) -> tuple[pd.DataFrame, tuple[pd.Timestamp, pd.Timestamp]]:
    start, end = build_session_window(df_jst, day_anchor_hhmm, sess_start_hhmm, sess_end_hhmm)
    # 包含フィルタ
    m = (df_jst["ts"] >= start) & (df_jst["ts"] <= end)
    sub = df_jst.loc[m].copy()

    # 空なら ±1h 緩和（運用保険）
    if sub.empty:
        alt_start = start - pd.Timedelta(hours=1)
        alt_end = end + pd.Timedelta(hours=1)
        m2 = (df_jst["ts"] >= alt_start) & (df_jst["ts"] <= alt_end)
        sub = df_jst.loc[m2].copy()
    return sub, (start, end)


def compute_change_vs_anchor(sub: pd.DataFrame, anchor_ts: pd.Timestamp) -> pd.Series:
    """
    アンカー時刻以降で最初の値を基準に % 変化を計算。
    無ければフォールバックで先頭値。
    """
    if sub.empty:
        return pd.Series(dtype=float)

    s = sub.loc[sub["ts"] >= anchor_ts, "value"]
    if s.empty:
        base = float(sub["value"].iloc[0])
    else:
        base = float(s.iloc[0])

    pct = (sub["value"].astype(float) / base - 1.0) * 100.0
    pct.name = "pct"
    return pct


# ---------- 描画（黒背景・白枠なし） ----------
def _apply_dark(ax):
    ax.set_facecolor("#111")
    ax.figure.set_facecolor("#111")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(True, color="#222")
    ax.tick_params(colors="#bbb")
    ax.yaxis.label.set_color("#bbb")
    ax.xaxis.label.set_color("#bbb")
    ax.title.set_color("#ddd")


def save_intraday_plot(df_ts_pct: pd.DataFrame, out_png: str, title: str, subtitle_time: pd.Timestamp):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    _apply_dark(ax)

    ax.plot(df_ts_pct["ts"], df_ts_pct["pct"], linewidth=2)
    ax.set_title(f"{title} Intraday Snapshot ({subtitle_time:%Y/%m/%d %H:%M})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Anchor (%)")

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_empty_plot(out_png: str, sess_start: pd.Timestamp, sess_end: pd.Timestamp):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    _apply_dark(ax)
    ax.text(
        0.5, 0.5, "No data in session", ha="center", va="center", transform=ax.transAxes, color="#888", fontsize=13
    )
    ax.set_title(f"Intraday Snapshot (No Data: {sess_start:%H:%M}-{sess_end:%H:%M})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Anchor (%)")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------- メイン ----------
def main():
    args = parse_args()

    # ログ（CI で表示）
    print("=== intraday snapshot ===")
    print(f"INDEX_KEY={args.index_key}")
    print(f"SESSION_START={args.session_start}")
    print(f"SESSION_END={args.session_end}")
    print(f"DAY_ANCHOR={args.day_anchor}")
    print(f"BASIS={args.basis}")

    # 入力
    df = load_intraday(args.csv, args.index_key)

    # セッションで切り出し
    sub, (sess_start, sess_end) = filter_session(df, args.session_start, args.session_end, args.day_anchor)
    anchor_ts = _at_day(args.day_anchor, sess_start.normalize())  # アンカーは同日 HH:MM

    now_jst = pd.Timestamp.now(tz=JST)

    if sub.empty:
        print("[WARN] セッション内データがありません。空の出力を生成します。")
        save_empty_plot(args.snapshot_png, sess_start, sess_end)

        stats = {
            "index_key": args.index_key,
            "label": args.index_key.upper(),
            "pct_intraday": None,
            "basis": args.basis,
            "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
            "updated_at": now_jst.isoformat(),
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(stats, ensure_ascii=False, indent=2))

        text = (
            f"▲ {args.index_key.upper()} 日中スナップショット（{now_jst:%Y/%m/%d %H:%M}）\n"
            f"※ データ無し（{args.session_start}–{args.session_end} / 基準: {args.basis}）\n"
            f"#{args.index_key.upper()} #日本株"
        )
        Path(args.out_text).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_text).write_text(text)
        return

    # ％変化（アンカー基準）
    pct_series = compute_change_vs_anchor(sub, anchor_ts)
    df_plot = pd.DataFrame({"ts": sub["ts"].values, "pct": pct_series.values})

    # グラフ
    save_intraday_plot(df_plot, args.snapshot_png, args.index_key.upper(), now_jst)

    # 最新値
    pct_last = float(pct_series.iloc[-1])

    # JSON
    stats = {
        "index_key": args.index_key,
        "label": args.index_key.upper(),
        "pct_intraday": pct_last,
        "basis": args.basis,
        "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
        "updated_at": now_jst.isoformat(),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    # TXT（ポスト用）
    post = (
        f"▲ {args.index_key.upper()} 日中スナップショット（{now_jst:%Y/%m/%d %H:%M}）\n"
        f"{pct_last:+.2f}%（基準: {args.basis}）\n"
        f"#{args.index_key.upper()} #日本株"
    )
    Path(args.out_text).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_text).write_text(post)


if __name__ == "__main__":
    main()
