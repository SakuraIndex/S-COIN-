#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d/7d/1m/1y) for Sakura Index series.
- セッション/タイムゾーンは INDEX_KEY に応じて自動切り替え
- 1d は 始値<終値: 青緑 / 始値>終値: 赤 / 同値: グレー
- 出来高があれば薄い棒で重ね描き
- データ不足の期間は自動スキップ（ガード）
出力先: docs/outputs/<index_key>_{1d|7d|1m|1y}.png
"""

import os
import re
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
#  基本設定
# ============================================================

OUTPUT_DIR = "docs/outputs"

# 色
COLOR_PRICE_DEFAULT = "#ff99cc"  # 長期線
COLOR_VOLUME = "#7f8ca6"
COLOR_UP = "#00C2A0"   # 陽線
COLOR_DOWN = "#FF4C4C" # 陰線
COLOR_EQUAL = "#CCCCCC"

plt.rcParams.update({
    "font.family": "Noto Sans CJK JP",
    "figure.facecolor": "#0b0f1a",
    "axes.facecolor": "#0b0f1a",
    "axes.edgecolor": "#27314a",
    "axes.labelcolor": "#e5ecff",
    "xtick.color": "#b8c2e0",
    "ytick.color": "#b8c2e0",
    "grid.color": "#27314a",
})

def log(msg: str):
    print(f"[long_charts] {msg}")

# ============================================================
#  市場セッション定義（INDEX_KEY で切替）
# ============================================================

def market_profile(index_key: str):
    k = (index_key or "").lower()

    # AIN-10：米国株 (ET 9:30-16:00 → JST表示)
    if k == "ain10":
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )

    # Astra4：米国株中心 (ET 9:30-16:00 → JST表示)
    if k == "astra4":
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )

    # S-COIN+：日本株 (JST 9:00-15:30)
    if k in ("scoin+", "scoin_plus", "scoinplus", "s-coin+"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 30),
        )

    # R-BANK9：日本株 (JST 9:00-15:00)
    if k == "rbank9":
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 0),
        )

    # fallback（JST 現物に準拠）
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        RAW_TZ_HISTORY="Asia/Tokyo",
        DISPLAY_TZ="Asia/Tokyo",
        SESSION_TZ="Asia/Tokyo",
        SESSION_START=(9, 0),
        SESSION_END=(15, 0),
    )

# ============================================================
#  入出力
# ============================================================

def _first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_intraday(base, key):
    return _first([
        f"{base}/{key}_intraday.csv",
        f"{base}/{key}_intraday.txt",
    ])

def find_history(base, key):
    return _first([
        f"{base}/{key}_history.csv",
        f"{base}/{key}_history.txt",
    ])

def parse_time_any(x, raw_tz, display_tz):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    # UNIX秒対応
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)
    # 汎用
    try:
        t = pd.to_datetime(s, errors="coerce")
        if pd.isna(t):
            return pd.NaT
        if t.tzinfo is None:
            t = t.tz_localize(raw_tz)
        return t.tz_convert(display_tz)
    except Exception:
        return pd.NaT

def pick_value_col(df):
    cols = [c.lower() for c in df.columns]
    for k in ["close", "price", "value", "index", "終値"]:
        if k in cols:
            return df.columns[cols.index(k)]
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[0] if nums else df.columns[0]

def pick_volume_col(df):
    cols = [c.lower() for c in df.columns]
    for k in ["volume", "vol", "出来高"]:
        if k in cols:
            return df.columns[cols.index(k)]
    return None

def read_any(path, raw_tz, display_tz):
    """
    列名を最初に正規化してから時刻/値/出来高の列を選ぶ（Datetime エラー対策）。
    """
    if not path:
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path)

    # 列名正規化（小文字化 + trim）
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 時刻列推定
    time_candidates_order = ["datetime", "time", "timestamp", "date"]
    tcol = None
    for name in time_candidates_order:
        if name in df.columns:
            tcol = name
            break
    if tcol is None:
        fuzzy = [c for c in df.columns if ("time" in c) or ("date" in c)]
        if fuzzy:
            tcol = fuzzy[0]
    if tcol is None:
        raise KeyError(f"No time-like column found. columns={list(df.columns)}")

    vcol = pick_value_col(df)
    volcol = pick_volume_col(df)

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if volcol else 0

    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def to_daily(df, display_tz):
    if df.empty:
        return df
    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(display_tz).dt.date
    g = d.groupby("date", as_index=False).agg({"value": "last", "volume": "sum"})
    g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(display_tz)
    return g[["time", "value", "volume"]]

# ============================================================
#  グラフ補助
# ============================================================

def format_time_axis(ax, mode, tz):
    if mode == "1d":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    else:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6, tz=tz)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

def apply_y_padding(ax, series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        ax.set_ylim(0, 1)
        return
    lo, hi = s.min(), s.max()
    pad = (hi - lo) * 0.08 if hi != lo else max(abs(lo) * 0.02, 0.5)
    ax.set_ylim(lo - pad, hi + pad)

def et_session_to_jst_frame(base_ts_jst, session_tz, display_tz, start_hm, end_hm):
    et = base_ts_jst.tz_convert(session_tz)
    et_date = et.date()
    start_et = pd.Timestamp(et_date.year, et_date.month, et_date.day,
                            start_hm[0], start_hm[1], tz=session_tz)
    end_et = pd.Timestamp(et_date.year, et_date.month, et_date.day,
                          end_hm[0], end_hm[1], tz=session_tz)
    return start_et.tz_convert(display_tz), end_et.tz_convert(display_tz)

# ===== 追加：ガード共通関数 ==================================

def enough_rows(df: pd.DataFrame, need: int) -> bool:
    """有効値の点数チェック"""
    if df is None or df.empty:
        return False
    n = len(df.dropna(subset=["time", "value"]))
    return n >= need

# ============================================================
#  描画本体
# ============================================================

def plot_df(df, index_key, label, mode, tz, frame=None):
    if df.empty:
        log(f"skip plot {index_key}_{label} (empty)")
        return

    # 1d の色分け
    if mode == "1d":
        open_price = df["value"].iloc[0]
        close_price = df["value"].iloc[-1]
        if close_price > open_price:
            color_line = COLOR_UP
        elif close_price < open_price:
            color_line = COLOR_DOWN
        else:
            color_line = COLOR_EQUAL
        lw = 2.2
    else:
        color_line = COLOR_PRICE_DEFAULT
        lw = 1.8

    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.3)

    # 出来高（あれば）
    if "volume" in df.columns and df["volume"].abs().sum() > 0:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"],
                width=0.9 if mode == "1d" else 0.8,
                color=COLOR_VOLUME, alpha=0.35, zorder=1, label="Volume")

    ax1.plot(df["time"], df["value"], color=color_line, lw=lw,
             solid_capstyle="round", label="Index", zorder=3)
    ax1.set_title(f"{index_key.upper()} ({label})", color="#ffb6c1")
    ax1.set_xlabel("Time" if mode == "1d" else "Date")
    ax1.set_ylabel("Index Value")

    format_time_axis(ax1, mode, tz)
    apply_y_padding(ax1, df["value"])

    if frame is not None:
        # フレームの健全性チェック（NaT/逆転を避ける）
        left, right = frame
        if pd.notna(left) and pd.notna(right) and left < right:
            ax1.set_xlim(frame)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = f"{OUTPUT_DIR}/{index_key}_{label}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    log(f"saved: {outpath} (color={color_line})")

# ============================================================
#  メイン
# ============================================================

def main():
    index_key = os.environ.get("INDEX_KEY", "").lower()
    if not index_key:
        raise SystemExit("ERROR: INDEX_KEY not set")

    MP = market_profile(index_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    intraday_path = find_intraday(OUTPUT_DIR, index_key)
    history_path = find_history(OUTPUT_DIR, index_key)

    intraday = read_any(intraday_path, MP["RAW_TZ_INTRADAY"], MP["DISPLAY_TZ"]) if intraday_path else pd.DataFrame(columns=["time","value","volume"])
    history  = read_any(history_path,  MP["RAW_TZ_HISTORY"],  MP["DISPLAY_TZ"]) if history_path  else pd.DataFrame(columns=["time","value","volume"])
    daily_all = to_daily(history if not history.empty else intraday, MP["DISPLAY_TZ"])

    # ---- 1d（セッションで切り出し）----
    df_1d = pd.DataFrame(columns=["time","value","volume"])
    frame_1d = None
    if not intraday.empty:
        last_ts = intraday["time"].max()
        start_jst, end_jst = et_session_to_jst_frame(
            last_ts, MP["SESSION_TZ"], MP["DISPLAY_TZ"],
            MP["SESSION_START"], MP["SESSION_END"]
        )
        mask = (intraday["time"] >= start_jst) & (intraday["time"] <= end_jst)
        df_1d = intraday.loc[mask].copy()
        frame_1d = (start_jst, end_jst)

    # ---- ガード（十分な点数がなければ描かない）----
    # 1d は最低 5点（約5本）あれば描画
    if enough_rows(df_1d, 5):
        plot_df(df_1d, index_key, "1d", "1d", MP["DISPLAY_TZ"], frame=frame_1d)
    else:
        log("skip 1d: not enough intraday rows")

    # ---- 7d / 1m / 1y（終値ベースの長期）----
    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])

    # 期間ごとの最低点数（必要に応じて調整）
    MIN_ROWS = {
        "7d": 4,   # おおむね4営業日
        "1m": 10,  # 10日以上
        "1y": 50,  # 50日以上
    }

    for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
        sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))].copy()
        if not enough_rows(sub, MIN_ROWS[label]):
            log(f"skip {label}: not enough daily rows (have={len(sub)})")
            continue
        plot_df(sub, index_key, label, "long", MP["DISPLAY_TZ"])

if __name__ == "__main__":
    main()
