#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d / 7d / 1m / 1y) for Sakura Index series.

特長:
- 1d は日本時間のセッション枠で必ず表示（データゼロでも枠固定）
- intraday の生時刻が UTC/JST どちらでも自動補正（枠内ヒット数で採用）
- 値がリターン/差分に見える場合は前日終値や BASE_LEVEL で絶対値に復元
"""

import os, re
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ===================== Theme =====================
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

# ===================== Profiles =====================

def norm_key(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", (s or "").lower()).strip("_")

def market_profile(index_key: str):
    k = norm_key(index_key)

    # US stocks (display JST)
    if k in ("ain10", "astra4"):
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )

    # JP stocks (S-COIN+)
    if k in ("scoin_plus", "scoin", "s_coin_plus", "s_coin"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",   # ← 誤っている可能性があれば自動で UTC 再読込
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 30),
        )

    # JP stocks (R-BANK9)
    if k in ("rbank9", "r_bank9"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 0),
        )

    # default JP
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        RAW_TZ_HISTORY="Asia/Tokyo",
        DISPLAY_TZ="Asia/Tokyo",
        SESSION_TZ="Asia/Tokyo",
        SESSION_START=(9, 0),
        SESSION_END=(15, 0),
    )

# ===================== IO helpers =====================

def _first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_intraday(base, key):
    k = norm_key(key)
    return _first([f"{base}/{k}_intraday.csv", f"{base}/{k}_intraday.txt"])

def find_history(base, key):
    k = norm_key(key)
    return _first([f"{base}/{k}_history.csv", f"{base}/{k}_history.txt"])

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

def guess_time_col(cols_norm):
    for c in ("datetime", "time", "timestamp", "date"):
        if c in cols_norm: return c
    for c in cols_norm:
        if "time" in c or "date" in c: return c
    return None

def pick_value_col(df, index_key):
    raw = list(df.columns)
    norm = [norm_key(c) for c in raw]
    keyn = norm_key(index_key)
    # 1) 近似名
    for r, n in zip(raw, norm):
        if keyn and (keyn in n or n in keyn):
            if pd.api.types.is_numeric_dtype(df[r]): return r
    # 2) 数値列の最大分散
    nums = [c for c in raw if pd.api.types.is_numeric_dtype(df[c])]
    if nums:
        var = {c: pd.to_numeric(df[c], errors="coerce").var() for c in nums}
        return max(nums, key=lambda c: (var[c] if pd.notna(var[c]) else -1))
    return raw[0]

def pick_volume_col(df):
    for c in df.columns:
        n = norm_key(c)
        if n in ("volume", "vol", "出来高"): return c
    return None

def read_any(path, raw_tz, display_tz, index_key):
    if not path:
        return pd.DataFrame(columns=["time", "value", "volume"])
    df = pd.read_csv(path)
    raw_cols = list(df.columns)
    norm_cols = [norm_key(c) for c in raw_cols]
    ren = dict(zip(raw_cols, norm_cols))
    dfn = df.rename(columns=ren)
    tcol = guess_time_col(norm_cols)
    if not tcol:
        raise KeyError(f"No time-like column. columns={raw_cols}")
    vcol = pick_value_col(df, index_key)
    vol  = pick_volume_col(df)

    out = pd.DataFrame()
    out["time"]  = dfn[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    out["volume"]= pd.to_numeric(df[vol], errors="coerce") if vol else 0
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def history_as_daily(history_df, display_tz):
    if history_df.empty: return history_df
    d = history_df.copy()
    d["time"] = pd.to_datetime(d["time"])
    d["time"] = d["time"].dt.tz_convert(display_tz) if getattr(d["time"].dt, "tz", None) else d["time"].dt.tz_localize(display_tz)
    if "volume" not in d.columns: d["volume"] = 0
    return d[["time", "value", "volume"]]

# ===================== Axis helpers =====================

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

def today_session_bounds(display_tz: str, session_tz: str, start_hm, end_hm):
    now = pd.Timestamp.now(tz=display_tz)
    sess_today = now.tz_convert(session_tz).date()
    s = pd.Timestamp(sess_today.year, sess_today.month, sess_today.day, start_hm[0], start_hm[1], tz=session_tz)
    e = pd.Timestamp(sess_today.year, sess_today.month, sess_today.day, end_hm[0], end_hm[1], tz=session_tz)
    return s.tz_convert(display_tz), e.tz_convert(display_tz)

# ===================== Plot =====================

def plot_df(df, index_key, label, mode, tz, frame=None):
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.3)

    if df.empty:
        ax1.text(0.5, 0.5, "No data", transform=ax1.transAxes, ha="center", va="center",
                 color="#9aa3bd", fontsize=22)
        ax1.set_title(f"{norm_key(index_key).upper()} ({label})", color="#ffb6c1")
        ax1.set_xlabel("Time" if mode == "1d" else "Date")
        ax1.set_ylabel("Index Value")
        if frame is not None: ax1.set_xlim(frame)
        format_time_axis(ax1, mode, tz)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{norm_key(index_key)}_{label}.png", dpi=180)
        plt.close()
        log(f"saved empty: {index_key}_{label}")
        return

    if mode == "1d":
        op, cl = float(df["value"].iloc[0]), float(df["value"].iloc[-1])
        color = COLOR_UP if cl > op else (COLOR_DOWN if cl < op else COLOR_EQUAL)
        lw = 2.2
    else:
        color, lw = COLOR_PRICE_DEFAULT, 1.8

    if "volume" in df.columns and pd.notna(df["volume"]).sum() > 0 and df["volume"].abs().sum() > 0:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"], width=0.9 if mode == "1d" else 0.8,
                color=COLOR_VOLUME, alpha=0.35, zorder=1, label="Volume")

    ax1.plot(df["time"], df["value"], color=color, lw=lw, solid_capstyle="round",
             label="Index", zorder=3)
    ax1.set_title(f"{norm_key(index_key).upper()} ({label})", color="#ffb6c1")
    ax1.set_xlabel("Time" if mode == "1d" else "Date")
    ax1.set_ylabel("Index Value")

    format_time_axis(ax1, mode, tz)
    apply_y_padding(ax1, df["value"])
    if frame is not None: ax1.set_xlim(frame)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = f"{OUTPUT_DIR}/{norm_key(index_key)}_{label}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    log(f"saved: {outpath}")

# ===================== Value restoration =====================

def looks_like_return_or_diff(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return False
    rng = s.max() - s.min()
    med = s.median()
    return (rng <= 20) and (abs(med) <= 15)

def prev_close_for(date_ts: pd.Timestamp, daily: pd.DataFrame) -> float | None:
    if daily.empty: return None
    d = daily.sort_values("time")
    day = date_ts.tz_convert(d["time"].iloc[0].tzinfo).date()
    prev = d[d["time"].dt.date < day]["value"]
    if prev.empty: return None
    return float(prev.iloc[-1])

def restore_absolute(df_1d: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if df_1d.empty: return df_1d
    if not looks_like_return_or_diff(df_1d["value"]):
        return df_1d

    base_level = float(os.environ.get("BASE_LEVEL", "100"))
    anchor = prev_close_for(df_1d["time"].iloc[-1], daily)
    if anchor is None:
        anchor = base_level
        log(f"restore: prev close not found -> use BASE_LEVEL={anchor}")

    s = pd.to_numeric(df_1d["value"], errors="coerce")
    if s.abs().max() <= 50 and s.abs().median() <= 15:
        restored = anchor * (1.0 + s / 100.0)  # % とみなす
    else:
        restored = anchor + s                   # 差分とみなす
    out = df_1d.copy()
    out["value"] = restored
    return out

# ===================== Main =====================

def main():
    index_key = os.environ.get("INDEX_KEY", "").strip()
    if not index_key:
        raise SystemExit("ERROR: INDEX_KEY not set")

    MP = market_profile(index_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    intraday_path = find_intraday(OUTPUT_DIR, index_key)
    history_path  = find_history(OUTPUT_DIR, index_key)

    # history
    history = read_any(history_path, MP["RAW_TZ_HISTORY"], MP["DISPLAY_TZ"], index_key) if history_path else pd.DataFrame()
    daily_all = history_as_daily(history, MP["DISPLAY_TZ"]) if not history.empty else pd.DataFrame()

    # 今日のセッション枠（JST表示）
    start_jst, end_jst = today_session_bounds(MP["DISPLAY_TZ"], MP["SESSION_TZ"], MP["SESSION_START"], MP["SESSION_END"])
    frame_1d = (start_jst, end_jst)

    # intraday: タイムゾーン自動補正（主設定→代替設定）
    intraday = pd.DataFrame()
    if intraday_path:
        primary_tz = MP["RAW_TZ_INTRADAY"]
        alt_tz = "UTC" if primary_tz != "UTC" else "Asia/Tokyo"

        intraday_p = read_any(intraday_path, primary_tz, MP["DISPLAY_TZ"], index_key)
        hit_p = ((intraday_p["time"] >= start_jst) & (intraday_p["time"] <= end_jst)).sum()

        intraday_a = read_any(intraday_path, alt_tz, MP["DISPLAY_TZ"], index_key)
        hit_a = ((intraday_a["time"] >= start_jst) & (intraday_a["time"] <= end_jst)).sum()

        intraday = intraday_p if hit_p >= hit_a else intraday_a
        chosen = primary_tz if hit_p >= hit_a else alt_tz
        log(f"intraday tz chosen: {chosen} (hits {hit_p} vs {hit_a}, frame JST {start_jst:%H:%M}-{end_jst:%H:%M})")

    # 1d 切り出し + 絶対値復元
    if not intraday.empty:
        m = (intraday["time"] >= start_jst) & (intraday["time"] <= end_jst)
        df_1d = intraday.loc[m].copy().reset_index(drop=True)
        if not df_1d.empty:
            df_1d = restore_absolute(df_1d, daily_all)
        else:
            log("1d: no rows in frame after tz selection")
            df_1d = pd.DataFrame(columns=["time", "value", "volume"])
    else:
        log("1d: intraday file missing or empty")
        df_1d = pd.DataFrame(columns=["time", "value", "volume"])

    plot_df(df_1d, index_key, "1d", "1d", MP["DISPLAY_TZ"], frame=frame_1d)

    # 7d/1m/1y
    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
    if not daily_all.empty:
        for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
            sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))]
            plot_df(sub, index_key, label, "long", MP["DISPLAY_TZ"])
    else:
        if not intraday.empty:
            d = intraday.copy()
            d["date"] = d["time"].dt.tz_convert(MP["DISPLAY_TZ"]).dt.date
            g = d.groupby("date", as_index=False).agg({"value": "last", "volume": "sum"})
            g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(MP["DISPLAY_TZ"])
            g = g[["time", "value", "volume"]].sort_values("time")
            for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
                sub = g[g["time"] >= (now - timedelta(days=days))]
                plot_df(sub, index_key, label, "long", MP["DISPLAY_TZ"])
        else:
            for label in ("7d", "1m", "1y"):
                plot_df(pd.DataFrame(), index_key, label, "long", MP["DISPLAY_TZ"])

if __name__ == "__main__":
    main()
