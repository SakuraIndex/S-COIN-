#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d/7d/1m/1y) for Sakura Index series.

- 出力ファイル名は key_slug(index_key) で正規化（AIN-10→ain10, S-COIN+→scoin_plus 等）
- 1d はセッション（INDEX_KEYごとに）JST 表示で切り出し
  → 空になった場合は直近6時間のフォールバックで必ず PNG を出力
- 長期（7d/1m/1y）は日足（終値）で生成
- 1d の色分け：上昇=青緑 (#00C2A0) / 下降=赤 (#FF4C4C) / 同値=灰

出力: docs/outputs/<slug>_{1d|7d|1m|1y}.png
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

# ---- key slug ---------------------------------------------------------------

def key_slug(index_key: str) -> str:
    """
    出力ファイル名用のスラッグを作成。
    """
    k = (index_key or "").lower().strip()
    # よくある別名を正規化
    mapping = {
        "ain-10": "ain10", "ain_10": "ain10",
        "s-coin+": "scoin_plus", "scoin+": "scoin_plus", "scoinplus": "scoin_plus",
        "r-bank9": "rbank9", "r_bank9": "rbank9",
    }
    if k in mapping:
        return mapping[k]
    # それ以外は英数字以外を '_' に
    slug = re.sub(r"[^a-z0-9]+", "_", k).strip("_")
    return slug or "index"

# ---- market profile ---------------------------------------------------------

def market_profile(index_key: str):
    k = key_slug(index_key)

    # AIN-10：米国株（表示は JST、セッションは NY）
    if k == "ain10":
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )

    # Astra4：米国株中心
    if k == "astra4":
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )

    # S-COIN+：日本株 9:00-15:30 JST
    if k == "scoin_plus":
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 30),
        )

    # R-BANK9：日本株 9:00-15:00 JST
    if k == "rbank9":
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 0),
        )

    # fallback：JST
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        RAW_TZ_HISTORY="Asia/Tokyo",
        DISPLAY_TZ="Asia/Tokyo",
        SESSION_TZ="Asia/Tokyo",
        SESSION_START=(9, 0),
        SESSION_END=(15, 0),
    )

# ---- IO helpers -------------------------------------------------------------

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
    # epoch 秒
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)
    # 汎用
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

# ---- plot helpers -----------------------------------------------------------

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

def session_frame(base_ts_display, session_tz, display_tz, start_hm, end_hm):
    """
    表示TZの基準時刻から、セッションTZの本日枠を表示TZに戻して返す。
    """
    stz = session_tz
    et = base_ts_display.tz_convert(stz)
    d = et.date()
    start = pd.Timestamp(d.year, d.month, d.day, start_hm[0], start_hm[1], tz=stz)
    end = pd.Timestamp(d.year, d.month, d.day, end_hm[0], end_hm[1], tz=stz)
    return start.tz_convert(display_tz), end.tz_convert(display_tz)

# ---- plotting ---------------------------------------------------------------

def plot_df(df, title_key, label, mode, tz, outpath, frame=None):
    # カラー決定
    if mode == "1d" and not df.empty:
        open_p = df["value"].iloc[0]; close_p = df["value"].iloc[-1]
        if close_p > open_p: color_line = COLOR_UP
        elif close_p < open_p: color_line = COLOR_DOWN
        else: color_line = COLOR_EQUAL
        lw = 2.2
    else:
        color_line = COLOR_PRICE_DEFAULT; lw = 1.8

    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.3)

    # volume（あれば）
    if not df.empty and df["volume"].abs().sum() > 0:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"],
                width=0.9 if mode == "1d" else 0.8,
                color=COLOR_VOLUME, alpha=0.35, zorder=1, label="Volume")

    # 値線（空でも軸・タイトルは描く）
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
    log(f"saved: {outpath}  size={len(df)}")

# ---- main -------------------------------------------------------------------

def main():
    index_key = os.environ.get("INDEX_KEY", "")
    if not index_key:
        raise SystemExit("ERROR: INDEX_KEY not set")
    slug = key_slug(index_key)
    MP = market_profile(index_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 入力
    p_i = find_intraday(OUTPUT_DIR, slug)
    p_h = find_history(OUTPUT_DIR, slug)
    log(f"read intraday: {p_i}")
    log(f"read history : {p_h}")

    intraday = read_any(p_i, MP["RAW_TZ_INTRADAY"], MP["DISPLAY_TZ"]) if p_i else pd.DataFrame()
    history  = read_any(p_h, MP["RAW_TZ_HISTORY"],  MP["DISPLAY_TZ"]) if p_h else pd.DataFrame()
    daily_all = to_daily(history if not history.empty else intraday, MP["DISPLAY_TZ"])

    # --- 1d ---
    df_1d = pd.DataFrame(); frame_1d = None
    if not intraday.empty:
        last_ts = intraday["time"].max()
        start_jst, end_jst = session_frame(
            last_ts, MP["SESSION_TZ"], MP["DISPLAY_TZ"],
            MP["SESSION_START"], MP["SESSION_END"]
        )
        mask = (intraday["time"] >= start_jst) & (intraday["time"] <= end_jst)
        df_1d = intraday.loc[mask].copy()
        frame_1d = (start_jst, end_jst)

        # フォールバック（窓が空なら直近 6h）
        if df_1d.empty:
            cutoff = last_ts - pd.Timedelta(hours=6)
            df_1d = intraday[intraday["time"] >= cutoff].copy()
            frame_1d = (cutoff, last_ts)
            log("1d window empty -> fallback to last 6h")
    else:
        log("intraday not found -> 1d will be empty but still saved")

    plot_df(df_1d, slug, "1d", "1d", MP["DISPLAY_TZ"], f"{OUTPUT_DIR}/{slug}_1d.png", frame=frame_1d)

    # --- long windows ---
    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
    for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
        sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))].copy()
        if sub.empty:
            log(f"skip {label}: no data window");  # それでも古い画像を残したいなら保存しない方が安全
            continue
        plot_df(sub, slug, label, "long", MP["DISPLAY_TZ"], f"{OUTPUT_DIR}/{slug}_{label}.png")

if __name__ == "__main__":
    main()
