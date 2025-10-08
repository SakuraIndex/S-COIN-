#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d / 7d / 1m / 1y) for Sakura Index series.

・INDEX_KEY に応じて市場セッション/タイムゾーンを自動切替
・1d は 始値<終値: 青緑 / 始値>終値: 赤 / 同値: グレー
・出来高があれば薄い棒を重ね描き
・%系（騰落率など）を検知したら履歴の終値で絶対値に復元して描画
・1d のセッション切り出しは堅牢化 + フォールバックで「No data」を回避

出力: docs/outputs/<index_key>_{1d|7d|1m|1y}.png
環境変数:
  INDEX_KEY  …  リポ名のキー（例: ain10 / astra4 / s-coin+ / rbank9 など）
"""

from __future__ import annotations

import os
import re
from datetime import timedelta
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
#  出力/配色・スタイル
# ============================================================

OUTPUT_DIR = "docs/outputs"

COLOR_PRICE_DEFAULT = "#ff99cc"  # 長期線（1d以外）
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


def log(msg: str) -> None:
    print(f"[long_charts] {msg}")


# ============================================================
#  市場プロファイル
# ============================================================

def market_profile(index_key: str) -> dict:
    k = (index_key or "").lower()

    # AIN-10：米国株 (ET 9:30-16:00 → JST 表示)
    if k == "ain10":
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )

    # Astra4：米国株中心 (ET 9:30-16:00 → JST)
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
    if k in ("rbank9", "r-bank9", "r_bank9"):
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
#  入出力ユーティリティ
# ============================================================

def _first(paths) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def find_intraday(base: str, key: str) -> Optional[str]:
    return _first([
        f"{base}/{key}_intraday.csv",
        f"{base}/{key}_intraday.txt",
    ])


def find_history(base: str, key: str) -> Optional[str]:
    return _first([
        f"{base}/{key}_history.csv",
        f"{base}/{key}_history.txt",
    ])


def parse_time_any(x, raw_tz: str, display_tz: str) -> pd.Timestamp | pd.NaT:
    """生 or 文字列 or UNIX秒 -> display_tz の Timestamp"""
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()

    # UNIX秒（10桁）
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)

    try:
        t = pd.to_datetime(s, errors="coerce")
        if pd.isna(t):
            return pd.NaT
        if t.tzinfo is None:
            t = t.tz_localize(raw_tz)
        return t.tz_convert(display_tz)
    except Exception:
        return pd.NaT


def pick_value_col(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    for k in ["close", "price", "value", "index", "終値"]:
        if k in cols:
            return df.columns[cols.index(k)]
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[0] if nums else df.columns[0]


def pick_volume_col(df: pd.DataFrame) -> Optional[str]:
    cols = [c.lower() for c in df.columns]
    for k in ["volume", "vol", "出来高"]:
        if k in cols:
            return df.columns[cols.index(k)]
    return None


def read_any(path: Optional[str], raw_tz: str, display_tz: str) -> pd.DataFrame:
    """
    列名を小文字化してから time/value/volume を抽出（Datetime エラー回避版）
    """
    if not path:
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 時刻列
    tcol = None
    for name in ["datetime", "time", "timestamp", "date"]:
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


def to_daily(df: pd.DataFrame, display_tz: str) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["date"] = d["time"].dt.tz_convert(display_tz).dt.date
    g = d.groupby("date", as_index=False).agg({"value": "last", "volume": "sum"})
    g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(display_tz)
    return g[["time", "value", "volume"]]


# ============================================================
#  値の正規化（%→絶対値）
# ============================================================

def percent_like(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    q95 = s.abs().quantile(0.95)
    span = (s.max() - s.min())
    # だいたい ±50% 以内に収まる揺れは % とみなす
    return (q95 <= 50) and (span <= 120)


def choose_baseline(history: pd.DataFrame, intraday: pd.DataFrame) -> float:
    """復元に使う基準値（直近日の終値）を選ぶ。なければ素直に100を返す。"""
    if not history.empty:
        base = float(pd.to_numeric(history["value"], errors="coerce").dropna().iloc[-1])
        if np.isfinite(base) and base != 0:
            return base
    if not intraday.empty:
        cand = float(pd.to_numeric(intraday["value"], errors="coerce").dropna().iloc[0])
        if np.isfinite(cand) and cand != 0:
            # 初値が%（小さい）なら 100 を返す、それ以外はその値
            return 100.0 if abs(cand) < 50 else cand
    return 100.0


def normalize_values(df: pd.DataFrame, base: float) -> pd.DataFrame:
    """df['value'] が %っぽければ base を用いて絶対値に復元"""
    if df.empty:
        return df
    out = df.copy()
    if percent_like(out["value"]):
        out["value"] = (1.0 + out["value"] / 100.0) * base
        log(f"normalize: percent-like -> absolute using base={base:.4f}")
    return out


# ============================================================
#  軸フォーマット / Y余白 / セッション窓
# ============================================================

def format_time_axis(ax, mode: str, tz: str) -> None:
    if mode == "1d":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    else:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6, tz=tz)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def apply_y_padding(ax, series: pd.Series) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        ax.set_ylim(0, 1)
        return
    lo, hi = s.min(), s.max()
    pad = (hi - lo) * 0.08 if hi != lo else max(abs(lo) * 0.02, 0.5)
    ax.set_ylim(lo - pad, hi + pad)


def compute_session_window(intra: pd.DataFrame, mp: dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    intraday の最新時刻を基準に、その日のセッション開始/終了の
    表示タイムゾーン（mp['DISPLAY_TZ']）の window を返す。
    """
    disp_tz = mp["DISPLAY_TZ"]
    sess_tz = mp["SESSION_TZ"]
    s_h, s_m = mp["SESSION_START"]
    e_h, e_m = mp["SESSION_END"]

    last_local = intra["time"].dt.tz_convert(disp_tz).iloc[-1]

    start_sess = pd.Timestamp(last_local.year, last_local.month, last_local.day,
                              s_h, s_m, tz=sess_tz)
    end_sess = pd.Timestamp(last_local.year, last_local.month, last_local.day,
                            e_h, e_m, tz=sess_tz)

    start = start_sess.tz_convert(disp_tz)
    end = end_sess.tz_convert(disp_tz)
    if end <= start:
        end = end + pd.Timedelta(days=1)
    return start, end


# ============================================================
#  描画
# ============================================================

def plot_df(df: pd.DataFrame, index_key: str, label: str, mode: str,
            tz: str, frame: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None) -> None:
    if df.empty:
        # 空でも「No data」を明示出力しておく（サイト側の見え方が安定）
        fig, ax1 = plt.subplots(figsize=(10, 4.8))
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"{index_key.upper()} ({label})", color="#ffb6c1")
        ax1.set_xlabel("Time" if mode == "1d" else "Date")
        ax1.set_ylabel("Index Value")
        format_time_axis(ax1, mode, tz)
        ax1.text(0.5, 0.5, "No data", transform=ax1.transAxes,
                 ha="center", va="center", color="#b8c2e0", fontsize=20, alpha=0.7)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        outpath = f"{OUTPUT_DIR}/{index_key}_{label}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=180)
        plt.close()
        log(f"saved: {outpath} (empty)")
        return

    # 1d は色分け
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

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.grid(True, alpha=0.3)

    # 出来高（あれば）
    if "volume" in df.columns and np.nan_to_num(df["volume"].values, nan=0.0).sum() > 0:
        ax2 = ax1.twinx()
        # 1d は見やすいように広めの棒幅
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
        ax1.set_xlim(frame)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = f"{OUTPUT_DIR}/{index_key}_{label}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    log(f"saved: {outpath}")


# ============================================================
#  メイン
# ============================================================

def main() -> None:
    index_key = os.environ.get("INDEX_KEY", "").lower()
    if not index_key:
        raise SystemExit("ERROR: INDEX_KEY not set")

    MP = market_profile(index_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    intraday_path = find_intraday(OUTPUT_DIR, index_key)
    history_path = find_history(OUTPUT_DIR, index_key)

    intraday_raw = read_any(intraday_path, MP["RAW_TZ_INTRADAY"], MP["DISPLAY_TZ"]) if intraday_path else pd.DataFrame()
    history_raw = read_any(history_path, MP["RAW_TZ_HISTORY"], MP["DISPLAY_TZ"]) if history_path else pd.DataFrame()

    # % -> 絶対値復元
    baseline = choose_baseline(history_raw, intraday_raw)
    intraday = normalize_values(intraday_raw, baseline)
    history = normalize_values(history_raw, baseline)

    # ---- 1d：堅牢にセッション切り出し（空ならフォールバック）----
    if not intraday.empty:
        start_win, end_win = compute_session_window(intraday, MP)
        mask = (intraday["time"] >= start_win) & (intraday["time"] <= end_win)
        df_1d = intraday.loc[mask].copy()

        if df_1d.empty:
            log(f"1d slice empty; fallback window. "
                f"start={start_win}, end={end_win}, "
                f"min={intraday['time'].min()}, max={intraday['time'].max()}")

            sess_minutes = (MP["SESSION_END"][0]*60 + MP["SESSION_END"][1]) - \
                           (MP["SESSION_START"][0]*60 + MP["SESSION_START"][1])
            fallback_end = intraday["time"].max()
            fallback_start = fallback_end - pd.Timedelta(minutes=int(max(sess_minutes, 60)))
            df_1d = intraday[(intraday["time"] >= fallback_start) &
                             (intraday["time"] <= fallback_end)].copy()
            frame_1d = (fallback_start, fallback_end)
        else:
            frame_1d = (start_win, end_win)
    else:
        df_1d = pd.DataFrame()
        frame_1d = None

    plot_df(df_1d, index_key, "1d", "1d", MP["DISPLAY_TZ"], frame=frame_1d)

    # ---- 7d / 1m / 1y：終値ベース（日足化）----
    source_for_daily = history if not history.empty else intraday
    daily_all = to_daily(source_for_daily, MP["DISPLAY_TZ"])

    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
    for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
        sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))]
        if sub.empty:
            log(f"skip plot {index_key}_{label} (no data window)")
            plot_df(pd.DataFrame(), index_key, label, "long", MP["DISPLAY_TZ"])
            continue
        plot_df(sub, index_key, label, "long", MP["DISPLAY_TZ"])


if __name__ == "__main__":
    main()
