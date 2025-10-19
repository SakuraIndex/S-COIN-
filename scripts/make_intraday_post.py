#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
from pathlib import Path
import pandas as pd

INDEX_KEY_DEFAULT = os.environ.get("INDEX_KEY", "scoin_plus")
OUT_DIR = Path("docs/outputs")

# 取引時間（JST前提・CSVもJST想定）
TRADING_START = os.environ.get("TRADING_START", "09:00")
TRADING_END   = os.environ.get("TRADING_END",   "15:30")

# パラメータ（株式インデックス想定：ボラ控えめ）
EPS = 5.0          # 分母の下限（指数レベル値の極小回避）
CLAMP_PCT = 30.0   # 表示の上限（±30%）

def iso_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z")

def read_intraday(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"CSV must have >= 2 columns: {csv_path}")
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    return df.dropna(subset=["ts", "val"]).sort_values("ts").reset_index(drop=True)

def session_mask(series: pd.Series) -> pd.Series:
    sh, sm = map(int, TRADING_START.split(":"))
    eh, em = map(int, TRADING_END.split(":"))
    s_ok = (series.dt.hour > sh) | ((series.dt.hour == sh) & (series.dt.minute >= sm))
    e_ok = (series.dt.hour < eh) | ((series.dt.hour == eh) & (series.dt.minute <= em))
    return s_ok & e_ok

def choose_open_baseline(df_day: pd.DataFrame):
    """9:00 寄り以降の最初の有効値を基準に"""
    df_sess = df_day.loc[session_mask(df_day["ts"])].copy()
    if df_sess.empty:
        return None, "no_session"
    # |val|>=EPS を優先（極小分母回避）
    cand = df_sess.loc[df_sess["val"].abs() >= EPS]
    if not cand.empty:
        return float(cand.iloc[0]["val"]), "open@09:00"
    # どうしても小さい場合は最初の値
    return float(df_sess.iloc[0]["val"]), "open@09:00_small"

def percent_change(first: float, last: float) -> float | None:
    try:
        denom = max(abs(float(first)), abs(float(last)), EPS)
        pct = (float(last) - float(first)) / denom * 100.0
        if pct > CLAMP_PCT: pct = CLAMP_PCT
        if pct < -CLAMP_PCT: pct = -CLAMP_PCT
        return pct
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", default=INDEX_KEY_DEFAULT)
    ap.add_argument("--csv", default=str(OUT_DIR / f"{INDEX_KEY_DEFAULT}_intraday.csv"))
    ap.add_argument("--out-json", default=str(OUT_DIR / f"{INDEX_KEY_DEFAULT}_stats.json"))
    ap.add_argument("--out-text", default=str(OUT_DIR / f"{INDEX_KEY_DEFAULT}_post_intraday.txt"))
    args = ap.parse_args()

    df = read_intraday(Path(args.csv))
    if df.empty:
        txt = f"{args.index_key.upper()} intraday: (no data)\n"
        Path(args.out_text).write_text(txt, encoding="utf-8")
        Path(args.out_json).write_text(json.dumps({
            "index_key": args.index_key, "pct_1d": None, "delta_level": None,
            "scale": "percent", "basis": "no_data", "updated_at": iso_now()
        }, ensure_ascii=False), encoding="utf-8")
        return

    # 当日かつ取引時間内へ
    the_day = df["ts"].dt.floor("D").iloc[-1]
    df_day = df[df["ts"].dt.floor("D") == the_day].copy()
    if df_day.empty:
        txt = f"{args.index_key.upper()} intraday: (no data)\n"
        Path(args.out_text).write_text(txt, encoding="utf-8")
        Path(args.out_json).write_text(json.dumps({
            "index_key": args.index_key, "pct_1d": None, "delta_level": None,
            "scale": "percent", "basis": "no_pct_col", "updated_at": iso_now()
        }, ensure_ascii=False), encoding="utf-8")
        return

    base, basis_note = choose_open_baseline(df_day)
    df_sess = df_day.loc[session_mask(df_day["ts"])].copy()

    if base is None or df_sess.empty:
        # 取引時間内データが無い日のフォールバック
        last_val = float(df_day.iloc[-1]["val"])
        pct_val = None
        delta_level = None
        first_ts, last_ts = df_day.iloc[0]["ts"], df_day.iloc[-1]["ts"]
        basis_note = "no_session"
    else:
        first_ts, last_ts = df_sess.iloc[0]["ts"], df_sess.iloc[-1]["ts"]
        last_val = float(df_sess.iloc[-1]["val"])
        delta_level = last_val - float(base)
        pct_val = percent_change(base, last_val)

    pct_str = "N/A" if pct_val is None else f"{pct_val:+.2f}%"
    delta_str = "N/A" if delta_level is None else f"{delta_level:+.6f}"
    txt = (
        f"{args.index_key.upper()} 1d: Δ={delta_str} (level) "
        f"A%={pct_str} (basis={basis_note} sess={TRADING_START}-{TRADING_END} "
        f"valid={first_ts}->{last_ts})\n"
    )
    Path(args.out_text).write_text(txt, encoding="utf-8")

    payload = {
        "index_key": args.index_key,
        "pct_1d": None if pct_val is None else float(pct_val),
        "delta_level": None if delta_level is None else float(delta_level),
        "scale": "percent",
        "basis": basis_note,
        "updated_at": iso_now(),
    }
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
