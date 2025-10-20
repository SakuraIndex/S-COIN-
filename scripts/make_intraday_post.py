#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_intraday_post.py

インデックスの intraday CSV から
- 取引時間（セッション）内のスナップショットを抽出
- 基準値に対する騰落率(%)とレベル差分(Δ)を算出
- SNS 用テキストと stats.json を出力

R-BANK9 / Astra4 / S-COIN+（日本株想定）で共通利用可能。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np


JST = "Asia/Tokyo"


# ---------- Utils ----------

def _now_jst_iso() -> str:
    return pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(JST).isoformat()


def _to_jst(ts: pd.Series) -> pd.Series:
    """UTC/naive -> JST（tz-aware）へ安全に変換。"""
    s = ts.copy()
    if s.dt.tz is None:
        # naive を UTC とみなし JST へ
        s = s.dt.tz_localize("UTC")
    # pandas<2.2 互換: tz_localize(errors=...) は未サポートなので使わない
    s = s.dt.tz_convert(JST)
    return s


def _first_numeric_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("数値列が見つかりません（level/value/close などの列が必要）")


def _pick_value_column(df: pd.DataFrame) -> str:
    for name in ["level", "value", "close", "price", "index", "idx"]:
        if name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
            return name
    return _first_numeric_column(df)


@dataclass
class SessionSpec:
    mode: str = "stock"          # "stock" | "24h"
    start: str = "09:00"
    end: str = "15:30"
    label: str = "09:00–15:30"

    @staticmethod
    def from_args(session: str, start: Optional[str], end: Optional[str]) -> "SessionSpec":
        if session == "24h":
            return SessionSpec(mode="24h", start="00:00", end="23:59", label="24h")
        st = start or "09:00"
        en = end or "15:30"
        return SessionSpec(mode="stock", start=st, end=en, label=f"{st}–{en}")


# ---------- Core ----------

def load_intraday(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # ts 列推定
    ts_col = None
    for c in ["ts", "timestamp", "time", "datetime"]:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError("時刻列(ts/timestamp/time/datetime)が見つかりません")

    # 日時パース
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).copy()
    df["ts"] = _to_jst(df[ts_col])  # 正規化して ts へ
    df = df.drop(columns=[c for c in [ts_col] if c != "ts"])

    val_col = _pick_value_column(df)
    df = df.rename(columns={val_col: "value"}).sort_values("ts").reset_index(drop=True)
    return df[["ts", "value"]]


def filter_session(
    df: pd.DataFrame, session: SessionSpec, ref_ts: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    if df.empty:
        return df
    if session.mode == "24h":
        return df

    # セッション日は「最新データのJST日付」
    last_ts = ref_ts or df["ts"].iloc[-1]
    day = last_ts.tz_convert(JST).date()

    st_h, st_m = [int(x) for x in session.start.split(":")]
    en_h, en_m = [int(x) for x in session.end.split(":")]

    st = pd.Timestamp(year=day.year, month=day.month, day=day.day,
                      hour=st_h, minute=st_m, tz=JST)
    en = pd.Timestamp(year=day.year, month=day.month, day=day.day,
                      hour=en_h, minute=en_m, tz=JST)

    return df[(df["ts"] >= st) & (df["ts"] <= en)].copy()


def pick_basis_value(
    df: pd.DataFrame,
    basis: str,
    session: SessionSpec,
) -> Tuple[pd.Timestamp, float, str]:
    """
    基準値の抽出。
    戻り値: (basis_ts, basis_value, basis_label)
    """
    if df.empty:
        raise ValueError("セッション内データが空です")

    if basis == "auto":
        basis = "open@09:00" if session.mode == "stock" else "first_any"

    if basis in ("open", "first_any"):
        row = df.iloc[0]
        return row["ts"], float(row["value"]), "first_any" if basis == "first_any" else "open"

    if basis == "open@09:00":
        # セッション開始時刻に最も近い（同時刻以降の）値
        st_h, st_m = [int(x) for x in session.start.split(":")]
        day = df["ts"].iloc[-1].tz_convert(JST).date()
        st = pd.Timestamp(year=day.year, month=day.month, day=day.day,
                          hour=st_h, minute=st_m, tz=JST)
        cand = df[df["ts"] >= st]
        row = cand.iloc[0] if not cand.empty else df.iloc[0]
        return row["ts"], float(row["value"]), "open@09:00"

    if basis == "stable@10:00":
        # 10:00 以降の最初の値（無ければ先頭）
        day = df["ts"].iloc[-1].tz_convert(JST).date()
        t10 = pd.Timestamp(year=day.year, month=day.month, day=day.day,
                           hour=10, minute=0, tz=JST)
        cand = df[df["ts"] >= t10]
        row = cand.iloc[0] if not cand.empty else df.iloc[0]
        return row["ts"], float(row["value"]), "stable@10:00"

    # デフォルト
    row = df.iloc[0]
    return row["ts"], float(row["value"]), "first_any"


def compute_change(basis_value: float, last_value: float) -> Tuple[float, float]:
    """
    Δ（レベル差分）と騰落率(％)を返す。
    """
    delta_level = float(last_value - basis_value)
    if basis_value == 0 or not np.isfinite(basis_value):
        pct = float("nan")
    else:
        pct = (last_value / basis_value - 1.0) * 100.0
    return delta_level, pct


def format_post_line(
    label: str,
    pct: float,
    delta_level: float,
    basis_label: str,
    session_label: str,
    ts_start: pd.Timestamp,
    ts_last: pd.Timestamp,
) -> str:
    pct_s = f"{pct:+.2f}%" if np.isfinite(pct) else "N/A"
    line = (
        f"{label} 1d: Δ={delta_level:+.6f} (level) "
        f"A%={pct_s} (basis={basis_label} sess={session_label} "
        f"valid={ts_start.strftime('%Y-%m-%d %H:%M:%S')}→{ts_last.strftime('%Y-%m-%d %H:%M:%S')})"
    )
    return line


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make intraday post & stats")
    p.add_argument("--index-key", required=True, help="インデックスキー（例: rbank9, scoin_plus, astra4）")
    p.add_argument("--csv", required=True, help="intraday CSV パス")
    p.add_argument("--out-json", required=True, help="stats.json パス")
    p.add_argument("--out-text", required=True, help="投稿テキスト .txt パス")

    p.add_argument(
        "--basis",
        default="auto",
        choices=["auto", "open", "open@09:00", "first_any", "stable@10:00"],
        help="基準の取り方"
    )

    p.add_argument(
        "--session",
        default="stock",
        choices=["stock", "24h"],
        help="セッションモード: 日本株(stock) or 24h"
    )
    p.add_argument("--session-start", default=None, help="stock 時の開始 (HH:MM), default=09:00")
    p.add_argument("--session-end", default=None, help="stock 時の終了 (HH:MM), default=15:30")
    p.add_argument("--label", default=None, help="表示ラベル（例: R-BANK9 / S-COIN+ / Astra-4）")
    p.add_argument("--day-anchor", default=None, help="未使用/将来拡張用(例 JST@09:00)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    label = args.label or args.index_key.upper()
    session = SessionSpec.from_args(args.session, args.session_start, args.session_end)

    df_all = load_intraday(args.csv)
    if df_all.empty:
        # 何も出来ないので N/A を出力
        write_text(
            args.out_text,
            f"{label} 1d: Δ=N/A (level) A%=N/A (basis=no_data sess={session.label} valid=N/A)"
        )
        write_json(
            args.out_json,
            {
                "index_key": args.index_key,
                "pct_1d": None,
                "delta_level": None,
                "scale": "percent",
                "basis": "no_data",
                "updated_at": _now_jst_iso(),
                "session": session.mode,
                "session_label": session.label,
            },
        )
        return

    df = filter_session(df_all, session)
    if df.empty:
        # セッション内データ無し
        ts_last_all = df_all["ts"].iloc[-1]
        ts_first_all = df_all["ts"].iloc[0]
        write_text(
            args.out_text,
            f"{label} 1d: Δ=N/A (level) A%=N/A (basis=no_session sess={session.label} "
            f"valid={ts_first_all.strftime('%Y-%m-%d %H:%M:%S')}→{ts_last_all.strftime('%Y-%m-%d %H:%M:%S')})"
        )
        write_json(
            args.out_json,
            {
                "index_key": args.index_key,
                "pct_1d": None,
                "delta_level": None,
                "scale": "percent",
                "basis": "no_session",
                "updated_at": _now_jst_iso(),
                "session": session.mode,
                "session_label": session.label,
            },
        )
        return

    basis_ts, basis_value, basis_label = pick_basis_value(df, args.basis, session)
    last_ts = df["ts"].iloc[-1]
    last_value = float(df["value"].iloc[-1])

    delta_level, pct = compute_change(basis_value, last_value)

    post_line = format_post_line(
        label=label,
        pct=pct,
        delta_level=delta_level,
        basis_label=basis_label,
        session_label=session.label,
        ts_start=df["ts"].iloc[0],
        ts_last=last_ts,
    )
    write_text(args.out_text, post_line)

    payload = {
        "index_key": args.index_key,
        "pct_1d": float(pct) if np.isfinite(pct) else None,
        "delta_level": float(delta_level) if np.isfinite(delta_level) else None,
        "scale": "percent",
        "basis": basis_label,
        "updated_at": _now_jst_iso(),
        "session": session.mode,
        "session_label": session.label,
        "basis_ts": basis_ts.isoformat(),
        "last_ts": last_ts.isoformat(),
    }
    write_json(args.out_json, payload)


if __name__ == "__main__":
    main()
