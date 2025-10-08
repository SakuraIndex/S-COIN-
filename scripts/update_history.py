#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append/refresh daily history from intraday for Sakura Index repos.

- CSV 列名を小文字化して自動検出（Datetime/time/timestamp/date などを許容）
- 値列は INDEX_KEY に一致する列を最優先で自動選択。無ければ最初の数値列
- インデックスごとの補正（スケール/バイアス/絶対値化）を長期チャートと統一
- 既存 history があれば最終日を更新／無ければ新規作成

入出力:
  intraday: docs/outputs/<index_key>_intraday.csv  (レガシー名も自動検出)
  history : docs/outputs/<index_key>_history.csv   (無ければ作成)
"""

import os
import re
from typing import Optional, Tuple
import pandas as pd

OUTPUT_DIR = "docs/outputs"

# ------------------------------------------------------------
# インデックス別プロファイル（長期チャートと整合）
# ------------------------------------------------------------
def market_profile(index_key: str):
    k = (index_key or "").lower()

    # AIN-10 / Astra4: 米国株（表示はJST）
    if k in ("ain10", "astra4"):
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            DISPLAY_TZ="Asia/Tokyo",
            FIX_SCALE=1.0,
            FIX_BIAS=0.0,
            ABSOLUTE=False,
        )

    # S-COIN+: 日本株（JST、絶対値化が必要）
    if k in ("scoin+", "scoin_plus", "scoinplus", "s-coin+"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            FIX_SCALE=1.0,
            FIX_BIAS=0.0,
            ABSOLUTE=True,
        )

    # R-BANK9: 日本株（JST）
    if k in ("rbank9", "r-bank9", "r_bank9"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            FIX_SCALE=1.0,
            FIX_BIAS=0.0,
            ABSOLUTE=False,
        )

    # 既定
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        DISPLAY_TZ="Asia/Tokyo",
        FIX_SCALE=1.0,
        FIX_BIAS=0.0,
        ABSOLUTE=False,
    )

# ------------------------------------------------------------
# 便利関数
# ------------------------------------------------------------
def log(msg: str):
    print(f"[update_history] {msg}")

def _first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_intraday(index_key: str) -> Optional[str]:
    key = (index_key or "").lower()
    # 公式名とレガシー名の両方をサーチ
    candidates = [
        f"{OUTPUT_DIR}/{key}_intraday.csv",
        f"{OUTPUT_DIR}/{key}_intraday.txt",
        # レガシー命名（例: scoin_plus_intraday.csv）
        f"{OUTPUT_DIR}/{key.replace('-', '_')}_intraday.csv",
        f"{OUTPUT_DIR}/{key.replace('-', '_')}_intraday.txt",
        # よくある別表記
        f"{OUTPUT_DIR}/scoin_plus_intraday.csv" if "scoin" in key else None,
        f"{OUTPUT_DIR}/rbank9_intraday.csv" if "rbank" in key else None,
        f"{OUTPUT_DIR}/ain10_intraday.csv" if "ain" in key else None,
        f"{OUTPUT_DIR}/astra4_intraday.csv" if "astra" in key else None,
    ]
    return _first([p for p in candidates if p])

def find_history(index_key: str) -> str:
    key = (index_key or "").lower()
    # history が無い場合はこのパスに作成
    return f"{OUTPUT_DIR}/{key}_history.csv"

def pick_time_col(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    # 優先順
    for name in ("datetime", "time", "timestamp", "date"):
        if name in cols:
            return df.columns[cols.index(name)]
    # あいまい一致
    for c in df.columns:
        lc = c.lower()
        if ("time" in lc) or ("date" in lc):
            return c
    raise KeyError(f"time-like column not found: columns={list(df.columns)}")

def pick_value_col(df: pd.DataFrame, index_key: str) -> str:
    cols = [c.lower() for c in df.columns]
    # インデックス名に近い列を最優先
    candidates = [
        index_key,
        index_key.replace("-", "_"),
        index_key.replace("_", "-"),
        index_key.replace("+", "plus"),
        index_key.replace("plus", "+"),
    ]
    for alias in candidates:
        if alias in cols:
            return df.columns[cols.index(alias)]
    # 一般的な候補
    for k in ("close", "price", "value", "index", "終値"):
        if k in cols:
            return df.columns[cols.index(k)]
    # 数値列の最初
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    # 見つからなければ最初の列
    return df.columns[0]

def parse_time_any(x, raw_tz: str, display_tz: str) -> pd.Timestamp:
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()

    # UNIX 秒 (10digits)
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)

    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    if t.tzinfo is None:
        t = t.tz_localize(raw_tz)
    return t.tz_convert(display_tz)

def read_intraday(path: str, raw_tz: str, display_tz: str, index_key: str) -> pd.DataFrame:
    """intraday を time/value/volume へ正規化"""
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path)
    # 列名を小文字化・trim
    df.columns = [str(c).strip().lower() for c in df.columns]

    tcol = pick_time_col(df)
    vcol = pick_value_col(df, index_key)
    volcol = None
    for k in ("volume", "vol", "出来高"):
        if k in df.columns:
            volcol = k
            break

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if volcol else 0
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def daily_from_intraday(intraday: pd.DataFrame, display_tz: str) -> pd.DataFrame:
    """日次終値 + 出来高合計"""
    if intraday.empty:
        return pd.DataFrame(columns=["time", "value", "volume"])
    d = intraday.copy()
    d["date"] = d["time"].dt.tz_convert(display_tz).dt.date
    g = d.groupby("date", as_index=False).agg({"value": "last", "volume": "sum"})
    g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(display_tz)
    return g[["time", "value", "volume"]]

def apply_fixes(df: pd.DataFrame, scale: float, bias: float, absolute: bool) -> pd.DataFrame:
    if df.empty:
        return df
    y = df.copy()
    y["value"] = y["value"] * scale + bias
    if absolute:
        y["value"] = y["value"].abs()
    return y

def ensure_history_header(index_key: str) -> Tuple[str, str]:
    """
    既存 history の値列名を返す。無ければ <INDEX_KEY_UPPER> として作成時に使用。
    戻り値: (history_path, value_col_name)
    """
    hist_path = find_history(index_key)
    if os.path.exists(hist_path):
        dfh = pd.read_csv(hist_path)
        cols = [c.strip() for c in dfh.columns]
        # 値列（Date 以外）を一つ選択
        for c in cols:
            if c.lower() != "date":
                return hist_path, c
    # 新規作成時の列名
    return hist_path, index_key.upper().replace("-", "_")

# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------
def main():
    index_key = os.environ.get("INDEX_KEY", "").lower()
    if not index_key:
        raise SystemExit("ERROR: INDEX_KEY not set")

    prof = market_profile(index_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    intraday_path = find_intraday(index_key)
    if not intraday_path:
        raise SystemExit(f"ERROR: intraday file not found for key={index_key}")

    log(f"load intraday: {intraday_path}")
    intraday = read_intraday(
        intraday_path, prof["RAW_TZ_INTRADAY"], prof["DISPLAY_TZ"], index_key
    )
    if intraday.empty:
        raise SystemExit("ERROR: intraday is empty after normalization")

    # 補正（スケール/バイアス/絶対値化）
    intraday = apply_fixes(intraday, prof["FIX_SCALE"], prof["FIX_BIAS"], prof["ABSOLUTE"])

    # 日次化（終値）
    daily = daily_from_intraday(intraday, prof["DISPLAY_TZ"])
    if daily.empty:
        raise SystemExit("ERROR: daily aggregation resulted in empty frame")

    # 既存 history を読み込み/なければ作る
    hist_path, value_col = ensure_history_header(index_key)

    if os.path.exists(hist_path):
        dfh = pd.read_csv(hist_path)
        dfh.columns = [c.strip() for c in dfh.columns]
        # 既存の Date を tz-naive の日付として扱い、重複日を上書き更新
        dfh["Date"] = pd.to_datetime(dfh["Date"], errors="coerce").dt.date
    else:
        dfh = pd.DataFrame(columns=["Date", value_col])

    # 直近日付を反映
    daily["Date"] = daily["time"].dt.tz_convert(prof["DISPLAY_TZ"]).dt.date
    dlast = daily[["Date", "value"]].rename(columns={"value": value_col}).dropna()

    # マージ（同一日があれば value を更新）
    merged = (
        pd.concat([dfh, dlast], ignore_index=True)
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    merged.to_csv(hist_path, index=False)
    log(f"updated history: {hist_path}  rows={len(merged)}  value_col={value_col}")

if __name__ == "__main__":
    main()
