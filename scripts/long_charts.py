#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d/7d/1m/1y) for Sakura Index series.

- INDEX_KEY によって市場セッション/タイムゾーンを自動切替
- 1d は 始値<終値: 青緑 / 始値>終値: 赤 / 同値: グレー
- 1d の値が %（前日比/始値比）っぽいときは前日終値から“絶対値”へ復元
  * 基準は以下の優先度で取得: LAST_CLOSE(env) > docs/outputs/{slug}_base.txt > daily history 最新終値
- 出来高があれば薄い棒で重ね描き（列が無い場合はスキップ）

出力: docs/outputs/<slug>_{1d|7d|1m|1y}.png
期待: docs/outputs/<slug>_{history,intraday}.{csv,png} は他ジョブが生成
"""

import os
import re
from datetime import timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# 基本設定
# ============================================================

OUTPUT_DIR = "docs/outputs"

# colors
COLOR_PRICE_DEFAULT = "#ff99cc"   # 長期線
COLOR_VOLUME        = "#7f8ca6"
COLOR_UP            = "#00C2A0"   # 陽線
COLOR_DOWN          = "#FF4C4C"   # 陰線
COLOR_EQUAL         = "#CCCCCC"

plt.rcParams.update({
    "font.family": "Noto Sans CJK JP",
    "figure.facecolor": "#0b0f1a",
    "axes.facecolor":   "#0b0f1a",
    "axes.edgecolor":   "#27314a",
    "axes.labelcolor":  "#e5ecff",
    "xtick.color":      "#b8c2e0",
    "ytick.color":      "#b8c2e0",
    "grid.color":       "#27314a",
})

def log(msg: str):
    print(f"[long_charts] {msg}")

def norm_slug(s: str) -> str:
    """INDEX_KEY をファイル名用に正規化（小文字・記号置換）。"""
    s = (s or "").strip().lower()
    s = s.replace("+", "_plus").replace("-", "_").replace(" ", "_")
    return s

# ============================================================
# 市場セッション定義（INDEX_KEY で切替）
# ============================================================

def market_profile(index_key: str) -> dict:
    k = (index_key or "").lower()

    # AIN-10: 米株 (ET 9:30-16:00 → JSTへ表示)
    if k == "ain10":
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY ="Asia/Tokyo",
            DISPLAY_TZ     ="Asia/Tokyo",
            SESSION_TZ     ="America/New_York",
            SESSION_START  =(9, 30),
            SESSION_END    =(16, 0),
        )

    # Astra4: 米株中心 (ET 9:30-16:00 → JSTへ表示)
    if k == "astra4":
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY ="Asia/Tokyo",
            DISPLAY_TZ     ="Asia/Tokyo",
            SESSION_TZ     ="America/New_York",
            SESSION_START  =(9, 30),
            SESSION_END    =(16, 0),
        )

    # S-COIN+: 日本株 (JST 9:00-15:30)
    if k in ("scoin+", "scoin_plus", "scoinplus", "s-coin+"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY ="Asia/Tokyo",
            DISPLAY_TZ     ="Asia/Tokyo",
            SESSION_TZ     ="Asia/Tokyo",
            SESSION_START  =(9, 0),
            SESSION_END    =(15, 30),
        )

    # R-BANK9: 日本株 (JST 9:00-15:00)
    if k in ("rbank9", "r-bank9", "r_bank9"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY ="Asia/Tokyo",
            DISPLAY_TZ     ="Asia/Tokyo",
            SESSION_TZ     ="Asia/Tokyo",
            SESSION_START  =(9, 0),
            SESSION_END    =(15, 0),
        )

    # fallback（JST）
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        RAW_TZ_HISTORY ="Asia/Tokyo",
        DISPLAY_TZ     ="Asia/Tokyo",
        SESSION_TZ     ="Asia/Tokyo",
        SESSION_START  =(9, 0),
        SESSION_END    =(15, 0),
    )

# ============================================================
# 入出力ユーティリティ
# ============================================================

def _first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_intraday(base: str, slug: str) -> Optional[str]:
    return _first([
        f"{base}/{slug}_intraday.csv",
        f"{base}/{slug}_intraday.txt",
    ])

def find_history(base: str, slug: str) -> Optional[str]:
    return _first([
        f"{base}/{slug}_history.csv",
        f"{base}/{slug}_history.txt",
    ])

def parse_time_any(x, raw_tz: str, display_tz: str) -> pd.Timestamp | pd.NaT:
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()

    # UNIX秒（10桁）
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)

    # 汎用
    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    if t.tzinfo is None:
        t = t.tz_localize(raw_tz)
    return t.tz_convert(display_tz)

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
    """列名を標準化後に time/value/volume を抽出。"""
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
        raise KeyError(f"No time-like column; cols={list(df.columns)}")

    vcol   = pick_value_col(df)
    volcol = pick_volume_col(df)

    out = pd.DataFrame()
    out["time"]   = df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    out["value"]  = pd.to_numeric(df[vcol], errors="coerce")
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
# スケール補正（% → 絶対値）
# ============================================================

def _try_load_last_close(slug: str, daily_all: pd.DataFrame) -> Optional[float]:
    """
    前日終値(絶対値)を取得:
      1) env LAST_CLOSE
      2) docs/outputs/{slug}_base.txt
      3) daily_all の最新値（ある程度のスケールがある場合）
    """
    # 1) 環境変数
    v = os.environ.get("LAST_CLOSE", "").strip()
    if v:
        try:
            x = float(v)
            if x > 0:
                log(f"last_close from env: {x}")
                return x
        except Exception:
            pass

    # 2) base file
    base_path = os.path.join(OUTPUT_DIR, f"{slug}_base.txt")
    if os.path.exists(base_path):
        try:
            with open(base_path, "r", encoding="utf-8") as f:
                x = float(f.read().strip())
            if x > 0:
                log(f"last_close from base file: {x}")
                return x
        except Exception:
            pass

    # 3) daily history
    if daily_all is not None and not daily_all.empty:
        try:
            x = float(pd.to_numeric(daily_all["value"], errors="coerce").dropna().iloc[-1])
            if x > 50:  # “指数の実値らしさ”の簡易閾値
                log(f"last_close from daily_all: {x}")
                return x
        except Exception:
            pass

    log("last_close not found")
    return None


def normalize_intraday_scale(intraday: pd.DataFrame,
                             daily_all: pd.DataFrame,
                             slug: str) -> pd.DataFrame:
    """
    intraday['value'] が % っぽい場合、前日終値から絶対値へ復元。
      ・max(|value|) ≤ 0.2 → 小数％とみなし ×100
      ・max(|value|) ≤ 5.0 → ％とみなす
    基準が取れなければスキップ（%のまま）。
    """
    if intraday.empty:
        return intraday

    s = pd.to_numeric(intraday["value"], errors="coerce")
    if s.isna().all():
        return intraday

    max_abs = float(s.abs().max())
    is_decimal_pct = max_abs <= 0.2
    is_pct         = max_abs <= 5.0

    if not (is_decimal_pct or is_pct):
        # 既に絶対値らしい
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
# グラフ補助
# ============================================================

def format_time_axis(ax, mode: str, tz: str):
    if mode == "1d":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    else:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6, tz=tz)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

def apply_y_padding(ax, series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        ax.set_ylim(0, 1)
        return
    lo, hi = float(s.min()), float(s.max())
    pad = (hi - lo) * 0.08 if hi != lo else max(abs(lo) * 0.02, 0.5)
    ax.set_ylim(lo - pad, hi + pad)

def session_frame_on_display(base_ts: pd.Timestamp,
                             session_tz: str, display_tz: str,
                             start_hm: Tuple[int, int],
                             end_hm: Tuple[int, int]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    表示TZの基準日時(base_ts)を、セッションTZの日付へ変換して
    セッションの始終時刻を display_tz へ戻す。
    """
    base_in_session = base_ts.tz_convert(session_tz)
    d = base_in_session.date()
    start_s = pd.Timestamp(d.year, d.month, d.day, start_hm[0], start_hm[1], tz=session_tz)
    end_s   = pd.Timestamp(d.year, d.month, d.day, end_hm[0],   end_hm[1],   tz=session_tz)
    return start_s.tz_convert(display_tz), end_s.tz_convert(display_tz)

# ============================================================
# 描画
# ============================================================

def plot_df(df: pd.DataFrame, slug: str, label: str, mode: str,
            tz: str, frame=None):
    if df.empty:
        log(f"skip plot {slug}_{label} (empty)")
        return

    # 1d は色分け
    if mode == "1d":
        open_price  = float(df["value"].iloc[0])
        close_price = float(df["value"].iloc[-1])
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
    try:
        if "volume" in df.columns and pd.to_numeric(df["volume"], errors="coerce").fillna(0).abs().sum() > 0:
            ax2 = ax1.twinx()
            ax2.bar(df["time"], df["volume"],
                    width=0.9 if mode == "1d" else 0.8,
                    color=COLOR_VOLUME, alpha=0.35, zorder=1, label="Volume")
    except Exception:
        pass

    ax1.plot(df["time"], df["value"], color=color_line, lw=lw,
             solid_capstyle="round", label="Index", zorder=3)
    ax1.set_title(f"{slug.upper()} ({label})", color="#ffb6c1")
    ax1.set_xlabel("Time" if mode == "1d" else "Date")
    ax1.set_ylabel("Index Value")

    format_time_axis(ax1, mode, tz)
    apply_y_padding(ax1, df["value"])
    if frame is not None:
        ax1.set_xlim(frame)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
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

    # 入力
    intraday_path = find_intraday(OUTPUT_DIR, slug)
    history_path  = find_history(OUTPUT_DIR,  slug)

    intraday = read_any(intraday_path, MP["RAW_TZ_INTRADAY"], MP["DISPLAY_TZ"]) if intraday_path else pd.DataFrame()
    history  = read_any(history_path,  MP["RAW_TZ_HISTORY"],  MP["DISPLAY_TZ"]) if history_path  else pd.DataFrame()
    daily_all = to_daily(history if not history.empty else intraday, MP["DISPLAY_TZ"])

    # 1d: セッションで切り出し & スケール補正
    if not intraday.empty:
        intraday = normalize_intraday_scale(intraday, daily_all, slug)

        last_ts = intraday["time"].max()
        start_disp, end_disp = session_frame_on_display(
            last_ts, MP["SESSION_TZ"], MP["DISPLAY_TZ"], MP["SESSION_START"], MP["SESSION_END"]
        )
        mask = (intraday["time"] >= start_disp) & (intraday["time"] <= end_disp)
        df_1d = intraday.loc[mask].copy()
        frame_1d = (start_disp, end_disp)
    else:
        df_1d = pd.DataFrame()
        frame_1d = None

    plot_df(df_1d, slug, "1d", "1d", MP["DISPLAY_TZ"], frame=frame_1d)

    # 7d / 1m / 1y: 日次シリーズから切り出し
    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
    for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
        sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))]
        if sub.empty:
            log(f"skip plot {slug}_{label} (no window)")
            continue
        plot_df(sub, slug, label, "long", MP["DISPLAY_TZ"])

if __name__ == "__main__":
    main()
