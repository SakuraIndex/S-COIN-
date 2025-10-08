#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d/7d/1m/1y) for Sakura Index series.

修正ポイント（2025-10-08）:
- タイムゾーン: 既に tz-aware なら tz_convert のみ。naive は raw_tz で tz_localize → display_tz。
- 1d 値が「前日終値比％」っぽい場合は、直近日次履歴の終値から絶対値に復元。
- 長期(7d/1m/1y): 日次履歴CSVから適切な数値列を厳格に選択（index_key と ‘close/終値’ を優先）。
- 日本株(JST)は 09:00-15:30 のフレームを明示。米株は ET 09:30-16:00 を JST表示に変換。
- 軽いデータ欠損やNaT混入でも “No data” 以外は極力描けるように防御。

出力: docs/outputs/<index_key>_{1d|7d|1m|1y}.png
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

# -------------------- market profiles --------------------
def market_profile(index_key: str):
    k = (index_key or "").lower()

    # AIN-10 / Astra4: 米株 ET→JST 表示
    if k in ("ain10", "astra4"):
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )

    # S-COIN+: 日本株 9:00-15:30
    if k in ("scoin+", "scoin_plus", "scoinplus", "s-coin+"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 30),
        )

    # R-BANK9: 日本株 9:00-15:00
    if k in ("rbank9", "r-bank9", "rbank"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 0),
        )

    # fallback: JST 現物と同様
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        RAW_TZ_HISTORY="Asia/Tokyo",
        DISPLAY_TZ="Asia/Tokyo",
        SESSION_TZ="Asia/Tokyo",
        SESSION_START=(9, 0),
        SESSION_END=(15, 0),
    )

# -------------------- IO helpers --------------------
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

# -------------------- parsing & normalization --------------------
_TIME_CANDIDATES = ["datetime", "time", "timestamp", "date", "日時"]

def ensure_tz(ts, raw_tz: str, display_tz: str):
    """ts を pandas.Timestamp に整形し display_tz に揃える。"""
    if pd.isna(ts):
        return pd.NaT
    t = pd.to_datetime(ts, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    # 既に tz-aware?
    if getattr(t, "tzinfo", None) is None:
        t = t.tz_localize(raw_tz)
    return t.tz_convert(display_tz)

def parse_time_any(x, raw_tz: str, display_tz: str):
    s = str(x).strip()
    # UNIX秒（10桁）
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)
    return ensure_tz(x, raw_tz, display_tz)

def pick_value_col(df: pd.DataFrame, index_key: str | None, prefer_close=True):
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    # index key 列を最優先（大文字小文字/記号差吸収）
    if index_key:
        norm = re.sub(r"[^a-z0-9]+", "", index_key.lower())
        for c in df.columns:
            c_norm = re.sub(r"[^a-z0-9]+", "", str(c).lower())
            if c_norm == norm:
                return c
    # close/終値 を次点
    for k in ["close", "終値", "last", "price", "value", "index"]:
        if k in cols_lower:
            return cols_lower[k]
    # 数値列のうち非単調ゼロでない列
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[0]
    # 最後の砦
    return df.columns[0]

def pick_volume_col(df: pd.DataFrame):
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    for k in ["volume", "vol", "出来高"]:
        if k in cols_lower:
            return cols_lower[k]
    return None

def read_any(path, raw_tz, display_tz, index_key: str | None):
    """CSV/TXT を読み、time/value/volume 列だけに正規化して返す。"""
    if not path:
        return pd.DataFrame(columns=["time", "value", "volume"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["time", "value", "volume"])

    # 列名はそのまま保持しつつ探索時のみ小文字化
    lower_map = {str(c).strip().lower(): c for c in df.columns}

    # time 列決定
    tcol = None
    for k in _TIME_CANDIDATES:
        if k in lower_map:
            tcol = lower_map[k]
            break
    if tcol is None:
        # time/date を含む最初の列
        for c in df.columns:
            lc = str(c).lower()
            if "time" in lc or "date" in lc or "日時" in lc:
                tcol = c
                break
    if tcol is None:
        raise KeyError(f"No time-like column. cols={list(df.columns)}")

    vcol = pick_value_col(df, index_key=index_key)
    volcol = pick_volume_col(df)

    out = pd.DataFrame()
    out["time"] = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if volcol else 0
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def to_daily(df: pd.DataFrame, display_tz: str):
    """time を display_tz の日付にまとめ、終値=last, volume=sum で集計。"""
    if df.empty:
        return df
    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(display_tz).dt.normalize()
    g = d.groupby("date", as_index=False).agg({"value": "last", "volume": "sum"})
    g = g.rename(columns={"date": "time"})
    return g[["time", "value", "volume"]]

def maybe_percent_to_abs(df_1d: pd.DataFrame, daily_hist: pd.DataFrame):
    """
    intraday 値が％変化（対前日終値）らしいときは絶対値に復元。
    目安:
      - 値の範囲が -30%〜+30% 程度に収まり（abs(max-min) < 60）
      - かつ平均絶対値が 30 未満
    復元基準は「直近履歴の終値（前日終値）」→なければ 1d の最初の絶対値っぽいもの。
    """
    if df_1d.empty:
        return df_1d

    vals = df_1d["value"].astype(float)
    val_range = vals.max() - vals.min()
    mean_abs = vals.abs().mean()

    looks_percent = (val_range < 60) and (mean_abs < 30)

    if not looks_percent:
        return df_1d  # もともと絶対値

    # 基準終値
    base = np.nan
    if not daily_hist.empty:
        base = float(daily_hist.sort_values("time")["value"].iloc[-1])  # 直近
    if np.isnan(base):
        # 1d の最初が “ほぼゼロではない & 100未満” ならそれを基準扱いに（最後の砦）
        first_v = float(vals.iloc[0])
        base = 100.0 if abs(first_v) < 1e-6 else max(abs(first_v), 100.0)

    abs_val = base * (1.0 + vals / 100.0)
    out = df_1d.copy()
    out["value"] = abs_val
    return out

# -------------------- axis helpers --------------------
def format_time_axis(ax, mode, tzname: str):
    if mode == "1d":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=tzname))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tzname))
    else:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6, tz=tzname)
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

def session_frame_jst(base_ts_jst, session_tz, display_tz, start_hm, end_hm):
    """base_ts_jst の日付における取引時間枠を display_tz で返す。"""
    if session_tz == display_tz:
        d = base_ts_jst.tz_convert(display_tz)
        start = pd.Timestamp(d.year, d.month, d.day, start_hm[0], start_hm[1], tz=display_tz)
        end = pd.Timestamp(d.year, d.month, d.day, end_hm[0], end_hm[1], tz=display_tz)
        return start, end
    # 例: ET セッション → JST 表示
    et = base_ts_jst.tz_convert(session_tz)
    et_d = et.date()
    start_et = pd.Timestamp(et_d.year, et_d.month, et_d.day, start_hm[0], start_hm[1], tz=session_tz)
    end_et = pd.Timestamp(et_d.year, et_d.month, et_d.day, end_hm[0], end_hm[1], tz=session_tz)
    return start_et.tz_convert(display_tz), end_et.tz_convert(display_tz)

# -------------------- plotting --------------------
def plot_df(df, index_key, label, mode, tzname, frame=None):
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.3)

    if df.empty:
        ax1.set_title(f"{index_key.upper()} ({label})", color="#ffb6c1")
        ax1.text(0.5, 0.5, "No data", color="#8b93a7", ha="center", va="center", transform=ax1.transAxes, fontsize=22)
        ax1.set_xlabel("Time" if mode == "1d" else "Date")
        ax1.set_ylabel("Index Value")
        format_time_axis(ax1, mode, tzname)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        outpath = f"{OUTPUT_DIR}/{index_key}_{label}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=180)
        plt.close()
        log(f"saved (empty): {outpath}")
        return

    # 1d は色分け（陽線/陰線）
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

    # 出来高（あれば）
    if "volume" in df and df["volume"].fillna(0).abs().sum() > 0:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"], width=0.9 if mode == "1d" else 0.8,
                color=COLOR_VOLUME, alpha=0.35, zorder=1, label="Volume")

    ax1.plot(df["time"], df["value"], color=color_line, lw=lw, solid_capstyle="round", zorder=3, label="Index")
    ax1.set_title(f"{index_key.upper()} ({label})", color="#ffb6c1")
    ax1.set_xlabel("Time" if mode == "1d" else "Date")
    ax1.set_ylabel("Index Value")

    format_time_axis(ax1, mode, tzname)
    apply_y_padding(ax1, df["value"])
    if frame is not None:
        ax1.set_xlim(frame)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = f"{OUTPUT_DIR}/{index_key}_{label}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    log(f"saved: {outpath}")

# -------------------- main --------------------
def main():
    index_key = os.environ.get("INDEX_KEY", "").lower()
    if not index_key:
        raise SystemExit("ERROR: INDEX_KEY not set")

    MP = market_profile(index_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    intraday_path = find_intraday(OUTPUT_DIR, index_key)
    history_path = find_history(OUTPUT_DIR, index_key)

    intraday = read_any(intraday_path, MP["RAW_TZ_INTRADAY"], MP["DISPLAY_TZ"], index_key) if intraday_path else pd.DataFrame()
    history = read_any(history_path, MP["RAW_TZ_HISTORY"], MP["DISPLAY_TZ"], index_key) if history_path else pd.DataFrame()
    daily_all = to_daily(history if not history.empty else intraday, MP["DISPLAY_TZ"])

    # ---- 1d: セッションで切り出し（JST表示）----
    if not intraday.empty:
        last_ts = intraday["time"].dropna().max()
        if pd.isna(last_ts):
            df_1d = pd.DataFrame()
            frame_1d = None
        else:
            start_jst, end_jst = session_frame_jst(
                last_ts, MP["SESSION_TZ"], MP["DISPLAY_TZ"], MP["SESSION_START"], MP["SESSION_END"]
            )
            mask = (intraday["time"] >= start_jst) & (intraday["time"] <= end_jst)
            df_1d = intraday.loc[mask].copy()
            # ％→絶対値 変換の可能性に対応
            df_1d = maybe_percent_to_abs(df_1d, daily_all)
            frame_1d = (start_jst, end_jst)
    else:
        df_1d = pd.DataFrame()
        frame_1d = None

    plot_df(df_1d, index_key, "1d", "1d", MP["DISPLAY_TZ"], frame=frame_1d)

    # ---- 7d / 1m / 1y: 日次履歴から描画（履歴なければ intraday を日次化）----
    base_daily = daily_all.copy()
    if base_daily.empty and not intraday.empty:
        base_daily = to_daily(intraday, MP["DISPLAY_TZ"])

    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"]).normalize()
    windows = [("7d", 7), ("1m", 31), ("1y", 365)]
    for label, days in windows:
        if base_daily.empty:
            plot_df(pd.DataFrame(), index_key, label, "long", MP["DISPLAY_TZ"])
            continue
        sub = base_daily[base_daily["time"] >= (now - timedelta(days=days))]
        sub = sub.sort_values("time")
        plot_df(sub, index_key, label, "long", MP["DISPLAY_TZ"])

if __name__ == "__main__":
    main()
