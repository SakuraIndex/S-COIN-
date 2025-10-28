#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Append daily *level* into scoin_plus_history.csv.

優先順位:
  1) docs/outputs/scoin_plus_stats.json の "level" (今日更新なら採用)
  2) docs/outputs/scoin_plus_intraday.csv 最終行から推定
     - 列名ヒントと値レンジでレベル/％/比率を判定
     - ％/比率なら前日レベルから (1+r) でレベルに復元

既に当日が入っていれば何もしません。
"""

from __future__ import annotations
import os, json, math
from pathlib import Path
import pandas as pd

INDEX_KEY = os.environ.get("INDEX_KEY", "scoin_plus")
OUT_DIR   = Path(os.environ.get("OUT_DIR", "docs/outputs"))
HIST_CSV  = OUT_DIR / f"{INDEX_KEY}_history.csv"
INTRA_CSV = OUT_DIR / f"{INDEX_KEY}_intraday.csv"
STATS_JSON= OUT_DIR / f"{INDEX_KEY}_stats.json"

TODAY_JST = pd.Timestamp.now(tz="Asia/Tokyo").normalize()

def _load_history() -> pd.DataFrame:
    if HIST_CSV.exists():
        df = pd.read_csv(HIST_CSV)
    else:
        df = pd.DataFrame(columns=["date","value"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)
    return df

def _save_history(df: pd.DataFrame) -> None:
    df = df.sort_values("date")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(HIST_CSV, index=False, date_format="%Y-%m-%d")

def _try_stats_level() -> tuple[float|None, pd.Timestamp|None]:
    if not STATS_JSON.exists():
        return None, None
    try:
        j = json.loads(STATS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    lvl = j.get("level")
    up  = j.get("updated_at")
    ts  = pd.to_datetime(up, errors="coerce") if up else None
    if isinstance(lvl,(int,float)) and math.isfinite(lvl):
        return float(lvl), ts
    return None, ts

def _last_intraday_numbers() -> tuple[pd.Timestamp|None, dict[str,float]]:
    if not INTRA_CSV.exists(): return None, {}
    df = pd.read_csv(INTRA_CSV)
    if df.shape[0] == 0 or df.shape[1] < 2: return None, {}
    last = df.tail(1)
    tscol = df.columns[0]
    ts = pd.to_datetime(last.iloc[0][tscol], errors="coerce")
    nums: dict[str,float] = {}
    for c in last.columns[1:]:
        try:
            v = float(last.iloc[0][c])
            if math.isfinite(v): nums[c] = v
        except Exception:
            pass
    return ts, nums

def _pick_level_from_intraday(nums: dict[str,float], prev_level: float|None) -> float|None:
    # 1) 列名でレベル候補を優先
    for key in ["level","index","close","price","value"]:
        for k,v in nums.items():
            if key in k.lower() and v > 0:
                return v
    # 2) 比率/％らしき列ならレベル化（prev_level が必要）
    for key in ["pct","percent","rtn","return","chg","change"]:
        for k,v in nums.items():
            if key in k.lower() and prev_level and prev_level > 0:
                if abs(v) <= 1.2:        # 0.05 (=+5%)
                    return prev_level * (1.0 + v)
                elif abs(v) < 50:        # 5 (=+5%)
                    return prev_level * (1.0 + v/100.0)
    # 3) 値レンジからの推定
    if nums:
        v = list(nums.values())[0]
        if prev_level and prev_level > 0 and 0 < abs(v) < 0.5:
            return prev_level * (1.0 + v)          # 比率推定
        if 5 <= abs(v) <= 10000:
            return v                                # レベルらしい
        if prev_level and prev_level > 0 and 0 < abs(v) < 50:
            return prev_level * (1.0 + v/100.0)     # ％推定
    return None

def main():
    hist = _load_history()
    if not hist.empty and (hist["date"].dt.normalize() == TODAY_JST).any():
        print("[append] already appended today, skip."); return

    # 1) stats.json 優先（今日の更新のみ採用）
    lvl, ts = _try_stats_level()
    if lvl and lvl > 0 and ts is not None and ts.tz_localize(None).normalize() == TODAY_JST:
        row = {"date": TODAY_JST.tz_localize(None), "value": float(lvl)}
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        _save_history(hist); print(f"[append] from stats.json: {row}"); return

    # 2) intraday 最終行から推定
    prev = float(hist.iloc[-1]["value"]) if not hist.empty else None
    _ts, nums = _last_intraday_numbers()
    lvl2 = _pick_level_from_intraday(nums, prev)
    if lvl2 and lvl2 > 0:
        row = {"date": TODAY_JST.tz_localize(None), "value": float(lvl2)}
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        _save_history(hist); print(f"[append] from intraday: {row}")
    else:
        print("[append] could not determine a valid level; no update.")

if __name__ == "__main__":
    main()
