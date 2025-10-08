#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long-term charts (1d/7d/1m/1y) generator with robust session/tz handling
and auto scale normalization for intraday (% -> absolute).

- 出力ファイル名は key_slug(index_key) で正規化（AIN-10→ain10, S-COIN+→scoin_plus …）
- 1d はセッション枠（INDEX_KEY ごと）で JST 表示
  * セッションに合っていなければ ALT_RAW_TZ（例: UTC）で再解釈して補正
  * 値が％らしければ最新終値を基準に“絶対値”へ復元
  * それでも空なら直近6時間フォールバック（軸はセッション枠を維持）
- 7d/1m/1y は日足（終値）で描画
- 1d 色分け：上昇=青緑 / 下降=赤 / 同値=灰
"""

import os
import re
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUTPUT_DIR = "docs/outputs"

# colors
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

# -----------------------------------------------------------------------------
# Slug (ファイル名の統一)
# -----------------------------------------------------------------------------
def key_slug(index_key: str) -> str:
    k = (index_key or "").lower().strip()
    mapping = {
        "ain-10": "ain10", "ain_10": "ain10",
        "s-coin+": "scoin_plus", "scoin+": "scoin_plus", "scoinplus": "scoin_plus",
        "r-bank9": "rbank9", "r_bank9": "rbank9",
    }
    if k in mapping:
        return mapping[k]
    return re.sub(r"[^a-z0-9]+", "_", k).strip("_") or "index"

# -----------------------------------------------------------------------------
# 市場プロファイル
# -----------------------------------------------------------------------------
def market_profile(index_key: str):
    slug = key_slug(index_key)

    if slug == "ain10":
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
            ALT_RAW_TZ="UTC",
        )

    if slug == "astra4":
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
            ALT_RAW_TZ="UTC",
        )

    if slug == "scoin_plus":
        # 日本株 9:00–15:30（JST）
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 30),
            ALT_RAW_TZ="UTC",
        )

    if slug == "rbank9":
        # 日本株 9:00–15:00（JST）
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 0),
            ALT_RAW_TZ="UTC",
        )

    # fallback：JST
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        RAW_TZ_HISTORY="Asia/Tokyo",
        DISPLAY_TZ="Asia/Tokyo",
        SESSION_TZ="Asia/Tokyo",
        SESSION_START=(9, 0),
        SESSION_END=(15, 0),
        ALT_RAW_TZ="UTC",
    )

# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def _first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_intraday(base, slug):
    return _first([f"{base}/{slug}_intraday.csv", f"{base}/{slug}_intraday.txt"])

def find_history(base, slug):
    return _first([f"{base}/{slug}_history.csv", f"{base}/{slug}_history.txt"])

def parse_time_any(x, raw_tz, display_tz):
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

def read_csv_as_timeseries(path, raw_tz, display_tz):
    if not path:
        return pd.DataFrame(columns=["time", "value", "volume"])
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 時刻列
    tcol = None
    for name in ["datetime", "time", "timestamp", "date"]:
        if name in df.columns:
            tcol = name; break
    if tcol is None:
        fuzzy = [c for c in df.columns if ("time" in c) or ("date" in c)]
        if fuzzy: tcol = fuzzy[0]
    if tcol is None:
        raise KeyError(f"No time-like column in {path}: cols={list(df.columns)}")

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

# -----------------------------------------------------------------------------
# セッション関連
# -----------------------------------------------------------------------------
def session_frame(base_ts_display, session_tz, display_tz, start_hm, end_hm):
    et = base_ts_display.tz_convert(session_tz)
    d = et.date()
    start = pd.Timestamp(d.year, d.month, d.day, start_hm[0], start_hm[1], tz=session_tz)
    end = pd.Timestamp(d.year, d.month, d.day, end_hm[0], end_hm[1], tz=session_tz)
    return start.tz_convert(display_tz), end.tz_convert(display_tz)

def fraction_in_session(df, display_tz, session_tz, start_hm, end_hm):
    if df.empty: return 0.0
    last_ts = df["time"].max()
    start, end = session_frame(last_ts, session_tz, display_tz, start_hm, end_hm)
    mask = (df["time"] >= start) & (df["time"] <= end)
    return float(mask.sum()) / float(len(df))

def ensure_session_aligned(path_intraday, MP):
    intraday = read_csv_as_timeseries(path_intraday, MP["RAW_TZ_INTRADAY"], MP["DISPLAY_TZ"]) \
               if path_intraday else pd.DataFrame()
    ratio = fraction_in_session(intraday, MP["DISPLAY_TZ"], MP["SESSION_TZ"],
                                MP["SESSION_START"], MP["SESSION_END"])
    log(f"session coverage with RAW_TZ={MP['RAW_TZ_INTRADAY']}: {ratio:.2f}")

    if ratio < 0.25 and path_intraday and MP.get("ALT_RAW_TZ"):
        intraday_alt = read_csv_as_timeseries(path_intraday, MP["ALT_RAW_TZ"], MP["DISPLAY_TZ"])
        ratio_alt = fraction_in_session(intraday_alt, MP["DISPLAY_TZ"], MP["SESSION_TZ"],
                                        MP["SESSION_START"], MP["SESSION_END"])
        log(f"session coverage with ALT_RAW_TZ={MP['ALT_RAW_TZ']}: {ratio_alt:.2f}")
        if ratio_alt > ratio:
            log("use ALT_RAW_TZ result (better alignment)")
            return intraday_alt
    return intraday

# -----------------------------------------------------------------------------
# 値スケール補正（％→絶対値）
# -----------------------------------------------------------------------------
def normalize_intraday_scale(intraday: pd.DataFrame, daily_all: pd.DataFrame) -> pd.DataFrame:
    """
    intraday['value'] が％っぽい場合に最新終値を基準として絶対値へ変換する。
    例）max(|v|) <= 0.2 → 小数％とみなして v*=100、max(|v|) <= 5 → ％とみなす。
    いずれも last_close * (1 + v/100) へ。
    """
    if intraday.empty or daily_all.empty:
        return intraday

    s = pd.to_numeric(intraday["value"], errors="coerce")
    if s.isna().all():
        return intraday

    max_abs = float(s.abs().max())
    last_close = float(pd.to_numeric(daily_all["value"], errors="coerce").dropna().iloc[-1])

    # 日足の終値が十分なスケール（例: > 50）で、intraday が小幅なら％と判定
    is_decimal_pct = max_abs <= 0.2
    is_pct = (max_abs <= 5.0)  # 5% 以内の振れなら％と仮定

    if last_close > 50 and (is_decimal_pct or is_pct):
        pct = s * (100.0 if is_decimal_pct else 1.0)
        abs_val = last_close * (1.0 + (pct / 100.0))
        out = intraday.copy()
        out["value"] = abs_val
        log(f"normalized intraday scale using last_close={last_close:.4f} "
            f"(detected {'decimal-pct' if is_decimal_pct else 'pct'})")
        return out

    # すでに絶対値と判断
    return intraday

# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
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
        ax.set_ylim(0, 1); return
    lo, hi = s.min(), s.max()
    pad = (hi - lo) * 0.08 if hi != lo else max(abs(lo) * 0.02, 0.5)
    ax.set_ylim(lo - pad, hi + pad)

def plot_df(df, title_key, label, mode, tz, outpath, frame=None):
    if mode == "1d" and not df.empty:
        open_p = df["value"].iloc[0]; close_p = df["value"].iloc[-1]
        color_line = COLOR_UP if close_p > open_p else (COLOR_DOWN if close_p < open_p else COLOR_EQUAL)
        lw = 2.2
    else:
        color_line = COLOR_PRICE_DEFAULT; lw = 1.8

    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.3)

    if not df.empty and df["volume"].abs().sum() > 0:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"],
                width=0.9 if mode == "1d" else 0.8,
                color=COLOR_VOLUME, alpha=0.35, zorder=1, label="Volume")

    if not df.empty:
        ax1.plot(df["time"], df["value"], color=color_line, lw=lw,
                 solid_capstyle="round", label="Index", zorder=3)

    ax1.set_title(f"{title_key.upper()} ({label})", color="#ffb6c1")
    ax1.set_xlabel("Time" if mode == "1d" else "Date")
    ax1.set_ylabel("Index Value")

    format_time_axis(ax1, mode, tz)
    apply_y_padding(ax1, df["value"] if not df.empty else pd.Series([0, 1]))
    if frame is not None:
        ax1.set_xlim(frame)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    log(f"saved: {outpath} size={len(df)}")

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    index_key = os.environ.get("INDEX_KEY", "")
    if not index_key:
        raise SystemExit("ERROR: INDEX_KEY not set")
    slug = key_slug(index_key)
    MP = market_profile(index_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    p_i = find_intraday(OUTPUT_DIR, slug)
    p_h = find_history(OUTPUT_DIR, slug)
    log(f"intraday: {p_i}")
    log(f"history : {p_h}")

    # intraday（自動セッション整合）
    intraday = ensure_session_aligned(p_i, MP)

    # history -> daily
    history = read_csv_as_timeseries(p_h, MP["RAW_TZ_HISTORY"], MP["DISPLAY_TZ"]) if p_h else pd.DataFrame()
    daily_all = to_daily(history if not history.empty else intraday, MP["DISPLAY_TZ"])

    # まず％→絶対値の自動補正を試みる（1d の値スケールの整合）
    intraday = normalize_intraday_scale(intraday, daily_all)

    # ---- 1d (セッション枠) ----
    df_1d = pd.DataFrame(); frame_1d = None
    if not intraday.empty:
        last_ts = intraday["time"].max()
        start_jst, end_jst = session_frame(last_ts, MP["SESSION_TZ"], MP["DISPLAY_TZ"],
                                           MP["SESSION_START"], MP["SESSION_END"])
        mask = (intraday["time"] >= start_jst) & (intraday["time"] <= end_jst)
        df_1d = intraday.loc[mask].copy()
        frame_1d = (start_jst, end_jst)

        if df_1d.empty:
            cutoff = last_ts - pd.Timedelta(hours=6)
            df_1d = intraday[intraday["time"] >= cutoff].copy()
            log("1d window empty -> fallback to last 6h (axis kept to session)")
    else:
        log("intraday not found -> 1d empty (will still save canvas)")

    plot_df(df_1d, slug, "1d", "1d", MP["DISPLAY_TZ"], f"{OUTPUT_DIR}/{slug}_1d.png", frame=frame_1d)

    # ---- 7d / 1m / 1y ----
    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
    for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
        sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))].copy()
        if sub.empty:
            log(f"skip {label}: no data window"); continue
        plot_df(sub, slug, label, "long", MP["DISPLAY_TZ"], f"{OUTPUT_DIR}/{slug}_{label}.png")

if __name__ == "__main__":
    main()
