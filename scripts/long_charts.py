#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d/7d/1m/1y) for Sakura Index series.
- 1d の値が % っぽい場合は前日終値から絶対値に補正。
"""

import os
import re
from datetime import timedelta
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# 基本設定
# ============================================================

OUTPUT_DIR = "docs/outputs"

COLOR_PRICE_DEFAULT = "#ff99cc"
COLOR_VOLUME = "#7f8ca6"
COLOR_UP = "#00C2A0"
COLOR_DOWN = "#FF4C4C"
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

def norm_slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("+", "_plus").replace("-", "_").replace(" ", "_")
    return s

# ============================================================
# 市場セッション定義
# ============================================================

def market_profile(index_key: str) -> dict:
    k = (index_key or "").lower()
    if k in ("ain10", "ain-10"):
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )
    if k in ("astra4",):
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
    if k in ("rbank9", "r-bank9", "r_bank9"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 0),
        )
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        RAW_TZ_HISTORY="Asia/Tokyo",
        DISPLAY_TZ="Asia/Tokyo",
        SESSION_TZ="Asia/Tokyo",
        SESSION_START=(9, 0),
        SESSION_END=(15, 0),
    )

# ============================================================
# ファイル読み込み
# ============================================================

def _first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_intraday(base, slug):
    return _first([f"{base}/{slug}_intraday.csv", f"{base}/{slug}_intraday.txt"])

def find_history(base, slug):
    return _first([f"{base}/{slug}_history.csv", f"{base}/{slug}_history.txt"])

def parse_time_any(x, raw_tz: str, display_tz: str) -> Union[pd.Timestamp, pd.NaT]:
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)
    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    if t.tzinfo is None:
        t = t.tz_localize(raw_tz)
    return t.tz_convert(display_tz)

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
    if not path:
        return pd.DataFrame(columns=["time", "value", "volume"])
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = None
    for name in ["datetime", "time", "timestamp", "date"]:
        if name in df.columns:
            tcol = name
            break
    if not tcol:
        fuzzy = [c for c in df.columns if "time" in c or "date" in c]
        tcol = fuzzy[0] if fuzzy else None
    if not tcol:
        raise KeyError("No time-like column found.")
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
# スケール補正
# ============================================================

def _try_load_last_close(slug, daily_all):
    v = os.environ.get("LAST_CLOSE", "").strip()
    if v:
        try:
            x = float(v)
            if x > 0:
                log(f"last_close from env: {x}")
                return x
        except:
            pass
    base_path = os.path.join(OUTPUT_DIR, f"{slug}_base.txt")
    if os.path.exists(base_path):
        try:
            with open(base_path, "r", encoding="utf-8") as f:
                x = float(f.read().strip())
            if x > 0:
                log(f"last_close from base file: {x}")
                return x
        except:
            pass
    if daily_all is not None and not daily_all.empty:
        try:
            x = float(pd.to_numeric(daily_all["value"], errors="coerce").dropna().iloc[-1])
            if x > 50:
                log(f"last_close from daily_all: {x}")
                return x
        except:
            pass
    log("last_close not found")
    return None

def normalize_intraday_scale(intraday, daily_all, slug):
    if intraday.empty:
        return intraday
    s = pd.to_numeric(intraday["value"], errors="coerce")
    if s.isna().all():
        return intraday
    max_abs = float(s.abs().max())
    is_decimal_pct = max_abs <= 0.2
    is_pct = max_abs <= 5.0
    if not (is_decimal_pct or is_pct):
        return intraday
    last_close = _try_load_last_close(slug, daily_all)
    if last_close is None:
        log("normalize skipped (no last_close)")
        return intraday
    pct = s * (100.0 if is_decimal_pct else 1.0)
    abs_val = last_close * (1.0 + (pct / 100.0))
    out = intraday.copy()
    out["value"] = abs_val
    log(f"normalized %→abs using last_close={last_close:.4f} "
        f"({'decimal-pct' if is_decimal_pct else 'pct'})")
    return out

# ============================================================
# プロット
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
    lo, hi = float(s.min()), float(s.max())
    pad = (hi - lo) * 0.08 if hi != lo else max(abs(lo) * 0.02, 0.5)
    ax.set_ylim(lo - pad, hi + pad)

def session_frame_on_display(base_ts, session_tz, display_tz, start_hm, end_hm):
    base_in_sess = base_ts.tz_convert(session_tz)
    d = base_in_sess.date()
    start = pd.Timestamp(d.year, d.month, d.day, start_hm[0], start_hm[1], tz=session_tz)
    end = pd.Timestamp(d.year, d.month, d.day, end_hm[0], end_hm[1], tz=session_tz)
    return start.tz_convert(display_tz), end.tz_convert(display_tz)

def plot_df(df, slug, label, mode, tz, frame=None):
    if df.empty:
        log(f"skip {slug}_{label} (empty)")
        return
    if mode == "1d":
        open_p = df["value"].iloc[0]
        close_p = df["value"].iloc[-1]
        color_line = COLOR_UP if close_p > open_p else (COLOR_DOWN if close_p < open_p else COLOR_EQUAL)
        lw = 2.2
    else:
        color_line, lw = COLOR_PRICE_DEFAULT, 1.8
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.3)
    if "volume" in df and df["volume"].abs().sum() > 0:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"], width=0.9, color=COLOR_VOLUME, alpha=0.35)
    ax1.plot(df["time"], df["value"], color=color_line, lw=lw)
    ax1.set_title(f"{slug.upper()} ({label})", color="#ffb6c1")
    ax1.set_xlabel("Time" if mode == "1d" else "Date")
    ax1.set_ylabel("Index Value")
    format_time_axis(ax1, mode, tz)
    apply_y_padding(ax1, df["value"])
    if frame:
        ax1.set_xlim(frame)
    outpath = f"{OUTPUT_DIR}/{slug}_{label}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    log(f"saved: {outpath}")

# ============================================================
# メイン
# ============================================================

def main():
    index_key = os.environ.get("INDEX_KEY", "").strip()
    if not index_key:
        raise SystemExit("ERROR: INDEX_KEY not set")
    slug = norm_slug(index_key)
    MP = market_profile(index_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    intraday_path = find_intraday(OUTPUT_DIR, slug)
    history_path = find_history(OUTPUT_DIR, slug)
    intraday = read_any(intraday_path, MP["RAW_TZ_INTRADAY"], MP["DISPLAY_TZ"]) if intraday_path else pd.DataFrame()
    history = read_any(history_path, MP["RAW_TZ_HISTORY"], MP["DISPLAY_TZ"]) if history_path else pd.DataFrame()
    daily_all = to_daily(history if not history.empty else intraday, MP["DISPLAY_TZ"])
    intraday = normalize_intraday_scale(intraday, daily_all, slug)
    if not intraday.empty:
        last_ts = intraday["time"].max()
        start_disp, end_disp = session_frame_on_display(
            last_ts, MP["SESSION_TZ"], MP["DISPLAY_TZ"], MP["SESSION_START"], MP["SESSION_END"]
        )
        mask = (intraday["time"] >= start_disp) & (intraday["time"] <= end_disp)
        df_1d = intraday.loc[mask].copy()
        frame_1d = (start_disp, end_disp)
    else:
        df_1d, frame_1d = pd.DataFrame(), None
    plot_df(df_1d, slug, "1d", "1d", MP["DISPLAY_TZ"], frame=frame_1d)
    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
    for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
        sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))]
        if sub.empty:
            continue
        plot_df(sub, slug, label, "long", MP["DISPLAY_TZ"])

if __name__ == "__main__":
    main()
