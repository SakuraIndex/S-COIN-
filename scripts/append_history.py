#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append today's close level into docs/outputs/<index>_history.csv
優先度:
 1) <index>_stats.json の "level"
 2) <index>_intraday.csv の最終値
"""

from __future__ import annotations
import os, json
from pathlib import Path
import pandas as pd
from datetime import datetime
import pytz

INDEX_KEY = os.environ.get("INDEX_KEY", "scoin_plus")
OUT_DIR   = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
HIST_CSV  = OUT_DIR / f"{INDEX_KEY}_history.csv"
STATS_JSON= OUT_DIR / f"{INDEX_KEY}_stats.json"
INTRA_CSV = OUT_DIR / f"{INDEX_KEY}_intraday.csv"
MARKET_TZ = os.environ.get("MARKET_TZ", "Asia/Tokyo")

def log(msg): print(f"[append] {msg}", flush=True)
def today_str(tzname: str) -> str:
    return datetime.now(pytz.timezone(tzname)).strftime("%Y-%m-%d")

def read_level_from_stats() -> float | None:
    try:
        if STATS_JSON.exists():
            j = json.loads(STATS_JSON.read_text())
            v = j.get("level", None)
            if v is None: return None
            v = float(v)
            if not pd.isna(v): return v
    except Exception as e:
        log(f"stats.json read err: {e}")
    return None

def read_level_from_intraday() -> float | None:
    try:
        if not INTRA_CSV.exists(): return None
        df = pd.read_csv(INTRA_CSV)
        if df.shape[1] < 2: return None
        s = pd.to_numeric(df[df.columns[1]], errors="coerce").dropna()
        if s.empty: return None
        return float(s.iloc[-1])
    except Exception as e:
        log(f"intraday.csv read err: {e}")
        return None

def load_history() -> pd.DataFrame:
    if HIST_CSV.exists():
        try:
            df = pd.read_csv(HIST_CSV)
            if df.shape[1] == 2: df.columns = ["date","value"]
            df["date"]  = pd.to_datetime(df["date"], errors="coerce").dt.date
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["date","value"]).drop_duplicates(subset=["date"], keep="last")
            return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            log(f"history read err: {e}")
    return pd.DataFrame(columns=["date","value"])

def save_history(df: pd.DataFrame):
    d = df.sort_values("date").copy()
    d["date"] = d["date"].astype(str)
    HIST_CSV.write_text(d.to_csv(index=False))

def main():
    today = today_str(MARKET_TZ)
    log(f"INDEX={INDEX_KEY} TZ={MARKET_TZ} today={today}")

    level = read_level_from_stats() or read_level_from_intraday()
    if level is None:
        log("no level found; skip append"); return

    df = load_history()
    dt = pd.to_datetime(today).date()
    df = df[df["date"] != dt]
    df = pd.concat([df, pd.DataFrame([{"date": dt, "value": float(level)}])], ignore_index=True)
    save_history(df)
    log(f"appended {today} -> {level}")

if __name__ == "__main__":
    main()
