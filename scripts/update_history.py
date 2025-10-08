# scripts/update_history.py
import os
import sys
import pandas as pd
from datetime import datetime, timezone

INDEX_KEY = os.environ.get("INDEX_KEY", "").lower()
if not INDEX_KEY:
    sys.exit("ERROR: INDEX_KEY not set")

OUT_DIR = os.environ.get("OUT_DIR", "docs/outputs")

intraday_csv = os.path.join(OUT_DIR, f"{INDEX_KEY}_intraday.csv")
history_csv  = os.path.join(OUT_DIR, f"{INDEX_KEY}_history.csv")

def read_intraday_last():
    if not os.path.exists(intraday_csv):
        sys.exit(f"ERROR: intraday csv not found: {intraday_csv}")

    df = pd.read_csv(intraday_csv)
    # 列名ゆれ吸収
    tcol = next((c for c in df.columns if str(c).lower() in ("datetime","time","timestamp")), None)
    vcol = next((c for c in df.columns if str(c).lower() == INDEX_KEY), None)
    if tcol is None or vcol is None:
        sys.exit(f"ERROR: columns not found in intraday: time={tcol}, value={vcol}")

    # 最終行の値（終値相当）
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce", utc=True).dt.tz_convert("Asia/Tokyo")
    df = df.dropna(subset=[tcol, vcol])
    if df.empty:
        sys.exit("ERROR: intraday has no valid rows")

    last_row = df.sort_values(tcol).iloc[-1]
    d = last_row[tcol].date().isoformat()
    v = pd.to_numeric(last_row[vcol], errors="coerce")
    if pd.isna(v):
        sys.exit("ERROR: last value is NaN")
    return d, float(v)

def upsert_history(date_str, value):
    # 既存historyを読み込み or 新規作成
    if os.path.exists(history_csv):
        h = pd.read_csv(history_csv)
    else:
        h = pd.DataFrame(columns=["Date", INDEX_KEY])

    # 型を揃える
    if "Date" not in h.columns:
        h["Date"] = []
    if INDEX_KEY not in h.columns:
        h[INDEX_KEY] = []

    # 既存日付の更新 or 追加
    if (h["Date"] == date_str).any():
        h.loc[h["Date"] == date_str, INDEX_KEY] = value
    else:
        h = pd.concat([h, pd.DataFrame([{"Date": date_str, INDEX_KEY: value}])], ignore_index=True)

    # 日付順にソートして最近400行だけ保持（任意）
    h["Date"] = pd.to_datetime(h["Date"], errors="coerce").dt.date
    h = h.dropna(subset=["Date"]).sort_values("Date")
    h = h.tail(400)

    # 保存
    h.to_csv(history_csv, index=False)
    print(f"[update_history] wrote: {history_csv} ({len(h)} rows)")

if __name__ == "__main__":
    d, v = read_intraday_last()
    upsert_history(d, v)
