#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d/7d/1m/1y) for Sakura Index series.

- セッション/タイムゾーンは INDEX_KEY に応じて自動切替
- intraday が % 変化っぽい場合は history 直近終値（なければ当日最初のティック）を基準に絶対値へ変換
- 1d は 始値<終値: 青緑 / 始値>終値: 赤 / 同値: グレー
- 出来高があれば薄い棒で重ね描き

出力: docs/outputs/<index_key>_{1d|7d|1m|1y}.png
"""

from __future__ import annotations
import os
import re
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =====================================================================
# 設定
# =====================================================================

OUTPUT_DIR = "docs/outputs"

# colors
COLOR_PRICE_DEFAULT = "#ff99cc"
COLOR_VOLUME = "#7f8ca6"
COLOR_UP = "#00C2A0"
COLOR_DOWN = "#FF4C4C"
COLOR_EQUAL = "#9aa3b2"
TITLE = "#ffb6c1"

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

# =====================================================================
# 市場プロファイル
# =====================================================================

def market_profile(index_key: str) -> dict:
    k = (index_key or "").lower()

    if k == "ain10":
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )

    if k == "astra4":
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )

    if k in ("scoin+", "scoin_plus", "scoinplus", "s-coin+"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 30),
        )

    if k == "rbank9":
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 0),
        )

    # fallback
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        RAW_TZ_HISTORY="Asia/Tokyo",
        DISPLAY_TZ="Asia/Tokyo",
        SESSION_TZ="Asia/Tokyo",
        SESSION_START=(9, 0),
        SESSION_END=(15, 0),
    )

# =====================================================================
# 入出力ユーティリティ
# =====================================================================

def _first(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def find_intraday(base, key):
    return _first([f"{base}/{key}_intraday.csv", f"{base}/{key}_intraday.txt"])

def find_history(base, key):
    return _first([f"{base}/{key}_history.csv", f"{base}/{key}_history.txt"])

def parse_time_any(x, raw_tz, display_tz):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    # UNIX seconds
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)
    # ISO or free-form
    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    if getattr(t, "tzinfo", None) is None:
        t = t.tz_localize(raw_tz)
    return t.tz_convert(display_tz)

def pick_value_col(df: pd.DataFrame) -> str:
    cols_l = [str(c).strip().lower() for c in df.columns]
    for k in ("close","price","value","index","終値"):
        if k in cols_l:
            return df.columns[cols_l.index(k)]
    # 指数固有名（例: ain10 / scoin_plus）
    for k in cols_l:
        if re.fullmatch(r"[a-z0-9_+\-]+", k) and pd.api.types.is_numeric_dtype(df[df.columns[cols_l.index(k)]]):
            return df.columns[cols_l.index(k)]
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[0] if nums else df.columns[0]

def pick_volume_col(df: pd.DataFrame) -> str | None:
    cols_l = [str(c).strip().lower() for c in df.columns]
    for k in ("volume","vol","出来高"):
        if k in cols_l:
            return df.columns[cols_l.index(k)]
    return None

def read_any(path, raw_tz, display_tz) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["time","value","volume"])
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    cols_l = [c.lower() for c in df.columns]

    # time col
    tcol = None
    for name in ("datetime","time","timestamp","date"):
        if name in cols_l:
            tcol = df.columns[cols_l.index(name)]
            break
    if tcol is None:
        fuzzy = [df.columns[i] for i,c in enumerate(cols_l) if ("time" in c or "date" in c)]
        if fuzzy: tcol = fuzzy[0]
    if tcol is None:
        raise KeyError(f"No time-like column. columns={list(df.columns)}")

    vcol = pick_value_col(df)
    volcol = pick_volume_col(df)

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if volcol else 0
    out = out.dropna(subset=["time","value"]).sort_values("time").reset_index(drop=True)
    return out

def to_daily(df: pd.DataFrame, display_tz: str) -> pd.DataFrame:
    if df.empty: return df
    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(display_tz).dt.date
    g = d.groupby("date", as_index=False).agg({"value":"last", "volume":"sum"})
    g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(display_tz)
    return g[["time","value","volume"]]

# =====================================================================
# 判定・変換
# =====================================================================

def looks_like_percent_change(series: pd.Series) -> bool:
    """0 近辺に推移し、±20% 以内に収まるなら % 変化とみなす"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return False
    rng = s.max() - s.min()
    near_zero_start = abs(s.iloc[0]) < 1.5  # 始値近辺が 0%
    within_range = (s.min() > -30) and (s.max() < 30)
    return (rng <= 40) and within_range and near_zero_start

def convert_pct_to_absolute(df_intra: pd.DataFrame, last_close: float | None) -> pd.DataFrame:
    """intraday % -> 絶対値。基準は history 直近終値、なければ当日最初のティックの『相対基準値』を 100 とみなす"""
    df = df_intra.copy()
    if df.empty: return df
    if last_close is None or not np.isfinite(last_close) or last_close <= 0:
        # 基準が無ければ、当日最初の「仮基準」を 100 としてレベル化（見た目の“折れ”を防ぐ）
        base = 100.0
    else:
        base = float(last_close)
    df["value"] = base * (1.0 + df["value"] / 100.0)
    return df

# =====================================================================
# 描画
# =====================================================================

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
        ax.set_ylim(0,1)
        return
    lo, hi = float(s.min()), float(s.max())
    pad = (hi - lo) * 0.08 if hi != lo else max(abs(lo)*0.02, 0.5)
    ax.set_ylim(lo - pad, hi + pad)

def et_session_to_jst_frame(base_ts_jst, session_tz, display_tz, start_hm, end_hm):
    et = base_ts_jst.tz_convert(session_tz)
    d = et.date()
    start = pd.Timestamp(d.year,d.month,d.day,start_hm[0],start_hm[1], tz=session_tz)
    end   = pd.Timestamp(d.year,d.month,d.day,end_hm[0],end_hm[1], tz=session_tz)
    return start.tz_convert(display_tz), end.tz_convert(display_tz)

def plot_df(df, index_key, label, mode, tz, frame=None):
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.3)

    if df.empty:
        ax1.text(0.5, 0.5, "No data", transform=ax1.transAxes, ha="center", va="center", color="#9aa3b2", fontsize=24)
        ax1.set_title(f"{index_key.upper()} ({label})", color=TITLE)
        ax1.set_xlabel("Time" if mode=="1d" else "Date")
        ax1.set_ylabel("Index Value")
        format_time_axis(ax1, mode, tz)
    else:
        # 1d の色分け
        if mode == "1d":
            open_p = df["value"].iloc[0]
            close_p = df["value"].iloc[-1]
            if close_p > open_p:
                color_line = COLOR_UP
            elif close_p < open_p:
                color_line = COLOR_DOWN
            else:
                color_line = COLOR_EQUAL
            lw = 2.2
        else:
            color_line = COLOR_PRICE_DEFAULT
            lw = 1.8

        # volume
        if "volume" in df.columns and pd.to_numeric(df["volume"], errors="coerce").fillna(0).abs().sum() > 0:
            ax2 = ax1.twinx()
            ax2.bar(df["time"], df["volume"], width=0.9 if mode=="1d" else 0.8,
                    color=COLOR_VOLUME, alpha=0.35, zorder=1, label="Volume")

        ax1.plot(df["time"], df["value"], color=color_line, lw=lw, solid_capstyle="round", label="Index", zorder=3)
        ax1.set_title(f"{index_key.upper()} ({label})", color=TITLE)
        ax1.set_xlabel("Time" if mode=="1d" else "Date")
        ax1.set_ylabel("Index Value")
        format_time_axis(ax1, mode, tz)
        apply_y_padding(ax1, df["value"])
        if frame is not None:
            ax1.set_xlim(frame)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = f"{OUTPUT_DIR}/{index_key}_{label}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    log(f"saved: {outpath}")

# =====================================================================
# メイン
# =====================================================================

def main():
    index_key = os.environ.get("INDEX_KEY", "").lower()
    if not index_key:
        raise SystemExit("ERROR: INDEX_KEY not set")

    MP = market_profile(index_key)

    intraday_path = find_intraday(OUTPUT_DIR, index_key)
    history_path  = find_history(OUTPUT_DIR, index_key)

    intraday = read_any(intraday_path, MP["RAW_TZ_INTRADAY"], MP["DISPLAY_TZ"]) if intraday_path else pd.DataFrame()
    history  = read_any(history_path,  MP["RAW_TZ_HISTORY"],  MP["DISPLAY_TZ"]) if history_path  else pd.DataFrame()
    last_close = float(history["value"].iloc[-1]) if (not history.empty and "value" in history) else None

    # --- intraday が % のようなら絶対化 ---
    if not intraday.empty and looks_like_percent_change(intraday["value"]):
        log("intraday looks like % change -> converting to absolute using history last close")
        intraday = convert_pct_to_absolute(intraday, last_close)

    # --- 1d フレーム ---
    if not intraday.empty:
        last_ts = intraday["time"].max()
        start_jst, end_jst = et_session_to_jst_frame(
            last_ts, MP["SESSION_TZ"], MP["DISPLAY_TZ"], MP["SESSION_START"], MP["SESSION_END"]
        )
        mask = (intraday["time"] >= start_jst) & (intraday["time"] <= end_jst)
        df_1d = intraday.loc[mask].copy()
        frame_1d = (start_jst, end_jst)
    else:
        df_1d = pd.DataFrame()
        frame_1d = None

    # --- 日足系列（history 優先、なければ intraday の日次化） ---
    if not history.empty:
        daily_all = history[["time","value","volume"]].copy()
    else:
        daily_all = to_daily(intraday, MP["DISPLAY_TZ"])

    # --- プロット ---
    plot_df(df_1d,      index_key, "1d", "1d",  MP["DISPLAY_TZ"], frame=frame_1d)

    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
    for label, days in [("7d",7), ("1m",31), ("1y",365)]:
        sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))]
        sub = sub.sort_values("time")
        plot_df(sub, index_key, label, "long", MP["DISPLAY_TZ"])

if __name__ == "__main__":
    main()
