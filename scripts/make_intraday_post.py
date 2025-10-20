#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate intraday "snapshot" chart and post text + stats json
for an index from its intraday CSV.

Expected CSV format:
    Datetime,<INDEX_KEY>
    2025-10-20 00:00:00, 2.28
    ...

Notes
- Datetime is assumed to be UTC (naive ok) and converted to Asia/Tokyo (JST).
- Session is specified in JST (e.g., 09:00–15:30).
- Day anchor aligns trading date (e.g., 09:00). Trading date := floor((ts_JST - anchor)).
- Basis examples: "open@09:00" (session open at given time) or "prev_close".
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

JST_TZNAME = "Asia/Tokyo"


# --------------------------
# Utilities
# --------------------------

def parse_hhmm(hhmm: str) -> time:
    """Parse 'HH:MM' -> datetime.time"""
    try:
        h, m = hhmm.split(":")
        return time(int(h), int(m))
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid time format: '{hhmm}' (expected HH:MM)")


def ensure_series_jst(ts_series: pd.Series) -> pd.Series:
    """
    Convert a datetime-like series to tz-aware JST.
    - If naive: treat as UTC then convert to JST
    - If already tz-aware: convert to JST
    """
    # to_datetime(..., utc=True) treats naive as UTC and keeps aware if tz present
    jst = pd.to_datetime(ts_series, utc=True).dt.tz_convert(JST_TZNAME)
    return jst


def trading_date_of(ts_jst: pd.Timestamp, anchor: time) -> datetime.date:
    """
    Compute trading date using 'day-anchor' in JST.
    trading_date := date( (ts - anchor) )
    """
    shifted = ts_jst - timedelta(hours=anchor.hour, minutes=anchor.minute)
    return shifted.date()


def build_session_range(now_jst: pd.Timestamp, anchor: time,
                        start_t: time, end_t: time) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Build today's session start/end in JST using day-anchor scheme.
    """
    trade_date = trading_date_of(now_jst, anchor)
    start_ts = pd.Timestamp.combine(trade_date, start_t).tz_localize(JST_TZNAME)
    end_ts = pd.Timestamp.combine(trade_date, end_t).tz_localize(JST_TZNAME)
    # もし end < start（夜間など）が必要ならここで翌日補正（今回は株式 09:00–15:30 を想定）
    return start_ts, end_ts


@dataclass
class Basis:
    kind: str  # "open" or "prev_close"
    at: Optional[time] = None  # open at HH:MM if kind == "open"

    @staticmethod
    def parse(text: str) -> "Basis":
        text = text.strip().lower()
        if text.startswith("open@"):
            at = parse_hhmm(text.split("@", 1)[1])
            return Basis(kind="open", at=at)
        if text in ("open", "open@09:00"):  # 旧来互換
            return Basis(kind="open", at=parse_hhmm("09:00"))
        if text in ("prev_close", "previous_close", "close_y"):
            return Basis(kind="prev_close", at=None)
        raise argparse.ArgumentTypeError(
            f"Invalid basis: '{text}'. Expected 'open@HH:MM' or 'prev_close'."
        )


# --------------------------
# Core
# --------------------------

def load_intraday(csv_path: str, index_key: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Datetime" not in df.columns:
        raise ValueError("CSV must contain 'Datetime' column")
    if index_key not in df.columns:
        raise ValueError(f"CSV must contain '{index_key}' column")

    df = df[["Datetime", index_key]].copy()
    df["ts_jst"] = ensure_series_jst(df["Datetime"])
    df = df.sort_values("ts_jst").reset_index(drop=True)
    return df


def filter_session(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    m = (df["ts_jst"] >= start_ts) & (df["ts_jst"] <= end_ts)
    return df.loc[m].copy()


def session_open_value(df_sess: pd.DataFrame, index_key: str, open_at: time) -> Optional[float]:
    """
    Pick the first value at/after open time inside the session.
    If exact time not available, take the earliest record in session.
    """
    if df_sess.empty:
        return None
    # 目安：open_at 以降の最初
    after_open = df_sess[df_sess["ts_jst"].dt.time >= open_at]
    if not after_open.empty:
        return float(after_open.iloc[0][index_key])
    # フォールバック：セッション先頭
    return float(df_sess.iloc[0][index_key])


def compute_stats(df_sess: pd.DataFrame, index_key: str, basis: Basis) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns: (delta_level, pct_change) where pct_change is A% (1d)
    """
    if df_sess.empty:
        return None, None

    last_val = float(df_sess.iloc[-1][index_key])

    if basis.kind == "open":
        open_val = session_open_value(df_sess, index_key, basis.at or parse_hhmm("09:00"))
        if open_val is None or open_val == 0:
            return None, None
        delta = last_val - open_val
        pct = (last_val / open_val - 1.0) * 100.0
        return delta, pct

    elif basis.kind == "prev_close":
        # 前日終値を「アンカー(例09:00)より前の最後」に近似（必要に応じて改善）
        # ここでは簡便に、セッション開始前の最後の値があれば採用
        # セッション開始時刻を推定（df_sessの最初の時刻）
        sess_start = df_sess.iloc[0]["ts_jst"]
        prev = df_sess[df_sess["ts_jst"] < sess_start]
        if prev.empty:
            return None, None
        prev_close = float(prev.iloc[-1][index_key])
        if prev_close == 0:
            return None, None
        delta = last_val - prev_close
        pct = (last_val / prev_close - 1.0) * 100.0
        return delta, pct

    return None, None


def format_post_line(index_key: str,
                     delta: Optional[float],
                     pct: Optional[float],
                     basis_label: str,
                     session_label: str,
                     valid_range_label: str) -> str:
    if delta is None or pct is None:
        return (f"{index_key.upper()} 1d: Δ=N/A (level) A%=N/A "
                f"(basis={basis_label} sess={session_label} valid={valid_range_label})")
    return (f"{index_key.upper()} 1d: Δ={delta:+.6f} (level) A%={pct:+.2f}% "
            f"(basis={basis_label} sess={session_label} valid={valid_range_label})")


def plot_snapshot(df_sess: pd.DataFrame, index_key: str, label: str,
                  snapshot_png: str, title_ts: pd.Timestamp,
                  pct_series: Optional[pd.Series]) -> None:
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    if pct_series is None:
        # 何らかの線は必要：レベル差分にする
        y = df_sess[index_key].astype(float).values
        ax.plot(df_sess["ts_jst"].dt.strftime("%H:%M"), y, label=label or index_key.upper())
        ax.set_ylabel("Index level")
    else:
        ax.plot(df_sess["ts_jst"].dt.strftime("%H:%M"), pct_series.values, label=label or index_key.upper())
        ax.set_ylabel("Change vs Prev Close (%)")

    ax.set_title(f"{label or index_key.upper()} Intraday Snapshot ({title_ts.strftime('%Y/%m/%d %H:%M')})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(snapshot_png, dpi=150)
    plt.close()


def compute_pct_series(df_sess: pd.DataFrame, index_key: str, basis: Basis) -> Optional[pd.Series]:
    if df_sess.empty:
        return None
    if basis.kind == "open":
        open_val = session_open_value(df_sess, index_key, basis.at or parse_hhmm("09:00"))
        if open_val in (None, 0):
            return None
        return (df_sess[index_key].astype(float) / float(open_val) - 1.0) * 100.0
    elif basis.kind == "prev_close":
        # 不確実なのでグラフ用には None を返す
        return None
    return None


# --------------------------
# Main
# --------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Make intraday snapshot & post for index")
    p.add_argument("--index-key", required=True, help="index key (CSV column and json key)")
    p.add_argument("--csv", required=True, help="path to intraday CSV")
    p.add_argument("--out-json", required=True, help="path to write stats json")
    p.add_argument("--out-text", required=True, help="path to write one-line post text")
    p.add_argument("--snapshot-png", help="path to write snapshot PNG")
    p.add_argument("--label", default=None, help="label for plot legend/title")
    p.add_argument("--session-start", type=parse_hhmm, default=parse_hhmm("09:00"),
                   help="JST session start HH:MM (default 09:00)")
    p.add_argument("--session-end", type=parse_hhmm, default=parse_hhmm("15:30"),
                   help="JST session end HH:MM (default 15:30)")
    p.add_argument("--day-anchor", type=parse_hhmm, default=parse_hhmm("09:00"),
                   help="Trading-day anchor HH:MM in JST (default 09:00)")
    p.add_argument("--basis", default="open@09:00",
                   help="calc basis: 'open@HH:MM' or 'prev_close' (default open@09:00)")

    args = p.parse_args(argv)

    index_key = args.index_key

    # Load data
    df = load_intraday(args.csv, index_key)

    # Now in JST (avoid tz_localize on aware ts)
    now_jst = pd.Timestamp.now(tz=JST_TZNAME)

    # Session range by day-anchor
    sess_start_ts, sess_end_ts = build_session_range(now_jst, args.day_anchor, args.session_start, args.session_end)
    session_label = f"{args.session_start.strftime('%H:%M')}-{args.session_end.strftime('%H:%M')}"

    # Filter session data
    df_sess = filter_session(df, sess_start_ts, sess_end_ts)

    # Basis
    basis = Basis.parse(args.basis)
    basis_label = (f"open@{basis.at.strftime('%H:%M')}" if basis.kind == "open" else "prev_close")

    # Stats
    delta, pct = compute_stats(df_sess, index_key, basis)

    # Output JSON
    updated_at_iso = now_jst.isoformat()
    if delta is None or pct is None:
        stats = {
            "index_key": index_key,
            "pct_1d": None,
            "delta_level": None,
            "scale": "percent",
            "basis": "no_session",
            "updated_at": updated_at_iso,
        }
    else:
        stats = {
            "index_key": index_key,
            "pct_1d": float(pct),
            "delta_level": float(delta),
            "scale": "percent",
            "basis": basis_label,
            "updated_at": updated_at_iso,
        }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False)

    # Output text (post)
    valid_label = f"{sess_start_ts.strftime('%Y-%m-%d %H:%M:%S')}->{sess_end_ts.strftime('%Y-%m-%d %H:%M:%S')}"
    line = format_post_line(index_key, delta, pct, basis_label, session_label, valid_label)
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write(line + "\n")

    # Snapshot PNG (optional)
    if args.snapshot_png:
        pct_series = compute_pct_series(df_sess, index_key, basis)
        plot_snapshot(
            df_sess=df_sess,
            index_key=index_key,
            label=args.label or index_key.upper(),
            snapshot_png=args.snapshot_png,
            title_ts=now_jst,
            pct_series=pct_series,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
