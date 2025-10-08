#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d / 7d / 1m / 1y) for Sakura Index series.

・INDEX_KEY から市場プロファイル（タイムゾーン/セッション）を自動選択
・1d は 始値<終値→青緑 / 始値>終値→赤 / 同値→グレー
・出来高列があれば淡色バーで重ね描き
・列名がバラついても安全に時刻/値/出来高を特定
・セッション切り出しで空になったら当日全体→残っている時間帯だけにフォールバック
・1d が「差分/リターン」っぽい場合は前日終値（history）から絶対値へ自動復元

出力: docs/outputs/<index_key>_{1d|7d|1m|1y}.png
"""

import os
import re
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==========================
# Theme / colors
# ==========================
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

# ==========================
# Profiles
# ==========================

def norm_key(s: str) -> str:
    """列名/キー用：英数字以外をアンダーラインにして小文字化"""
    return re.sub(r"[^0-9a-z]+", "_", (s or "").lower()).strip("_")

def market_profile(index_key: str):
    k = norm_key(index_key)

    # AIN-10 / Astra4 → 米国株（ET 9:30-16:00 → JST 表示）
    if k in ("ain10", "astra4"):
        return dict(
            RAW_TZ_INTRADAY="America/New_York",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="America/New_York",
            SESSION_START=(9, 30),
            SESSION_END=(16, 0),
        )

    # S-COIN+ → 日本（JST 9:00-15:30）
    if k in ("scoin_plus", "scoin", "s_coin_plus", "s_coin"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 30),
        )

    # R-BANK9 → 日本（JST 9:00-15:00）
    if k in ("rbank9", "r_bank9"):
        return dict(
            RAW_TZ_INTRADAY="Asia/Tokyo",
            RAW_TZ_HISTORY="Asia/Tokyo",
            DISPLAY_TZ="Asia/Tokyo",
            SESSION_TZ="Asia/Tokyo",
            SESSION_START=(9, 0),
            SESSION_END=(15, 0),
        )

    # fallback: 日本現物
    return dict(
        RAW_TZ_INTRADAY="Asia/Tokyo",
        RAW_TZ_HISTORY="Asia/Tokyo",
        DISPLAY_TZ="Asia/Tokyo",
        SESSION_TZ="Asia/Tokyo",
        SESSION_START=(9, 0),
        SESSION_END=(15, 0),
    )

# ==========================
# IO helpers
# ==========================

def _first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_intraday(base, key):
    k = norm_key(key)
    return _first([
        f"{base}/{k}_intraday.csv",
        f"{base}/{k}_intraday.txt",
        # 既存命名の揺れに備える（例: scoin_plus → scoin_plus_intraday.csv は OK）
        # 明示的な旧ファイル名がある場合はここに追加
    ])

def find_history(base, key):
    k = norm_key(key)
    return _first([
        f"{base}/{k}_history.csv",
        f"{base}/{k}_history.txt",
    ])

def parse_time_any(x, raw_tz, display_tz):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    # UNIX 秒
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)
    # 通常
    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    if t.tzinfo is None:
        t = t.tz_localize(raw_tz)
    return t.tz_convert(display_tz)

def guess_time_col(cols_norm):
    # 優先順で時刻列（正規化済み名）を推定
    for c in ("datetime", "time", "timestamp", "date"):
        if c in cols_norm:
            return c
    # それっぽいもの
    for c in cols_norm:
        if "time" in c or "date" in c:
            return c
    return None

def pick_value_col(df, index_key):
    """
    値列を推定（優先：指数名に近い列 → 数値列のうち最大分散の列）
    """
    cols_raw = list(df.columns)
    cols_norm = [norm_key(c) for c in cols_raw]
    key_norm = norm_key(index_key)

    # 1) キーに近い候補（例: "s-coin+" → "s_coin_" などの揺れにも対応）
    candidates = []
    for raw, nn in zip(cols_raw, cols_norm):
        if key_norm and (key_norm in nn or nn in key_norm):
            candidates.append(raw)
    # plus/マイナスなどの別名も拾う
    alias_tokens = set(filter(None, re.split(r"[_]+", key_norm)))
    for raw, nn in zip(cols_raw, cols_norm):
        toks = set(filter(None, re.split(r"[_]+", nn)))
        if len(alias_tokens & toks) >= max(1, len(alias_tokens) - 1):
            candidates.append(raw)

    # 2) 数値列
    numeric_cols = [c for c in cols_raw if pd.api.types.is_numeric_dtype(df[c])]

    if candidates:
        # 数値である候補を優先
        for c in candidates:
            if c in numeric_cols:
                return c
        # どうしても数値でない場合は最初の数値列
        if numeric_cols:
            return numeric_cols[0]

    # 3) 分散が最も大きい数値列（リターン/出来高混在時の誤選択を減らす）
    if numeric_cols:
        variances = {c: pd.to_numeric(df[c], errors="coerce").var() for c in numeric_cols}
        return sorted(variances, key=lambda x: (variances[x] if pd.notna(variances[x]) else -1), reverse=True)[0]

    # 最後の手段
    return cols_raw[0]

def pick_volume_col(df):
    cols_norm = [norm_key(c) for c in df.columns]
    for raw, nn in zip(df.columns, cols_norm):
        if nn in ("volume", "vol", "出来高"):
            return raw
    return None

def read_any(path, raw_tz, display_tz, index_key):
    """列名を正規化してから安全に時刻/値/出来高を抽出"""
    if not path:
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path)
    # 列名の前処理
    original_cols = list(df.columns)
    cols_norm = [norm_key(c) for c in original_cols]
    col_map = dict(zip(original_cols, cols_norm))
    df_ren = df.rename(columns=col_map)

    tcol_n = guess_time_col(cols_norm)
    if not tcol_n:
        raise KeyError(f"No time-like column. columns={original_cols}")

    vcol_raw = pick_value_col(df, index_key)
    vol_raw = pick_volume_col(df)

    # 時刻を tz-aware へ
    out = pd.DataFrame()
    out["time"] = df_ren[tcol_n].apply(lambda x: parse_time_any(x, raw_tz, display_tz))

    # 値/出来高
    out["value"] = pd.to_numeric(df[vcol_raw], errors="coerce")
    out["volume"] = pd.to_numeric(df[vol_raw], errors="coerce") if vol_raw else 0

    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def to_daily_from_history(history_df, display_tz):
    """history は既に日次想定。互換のため最低限の列整形のみ。"""
    if history_df.empty:
        return history_df
    d = history_df.copy()
    # 念のため tz/localize
    if "time" not in d.columns:
        return pd.DataFrame(columns=["time", "value", "volume"])
    d["time"] = pd.to_datetime(d["time"]).dt.tz_convert(display_tz) if getattr(d["time"].dt, "tz", None) else pd.to_datetime(d["time"]).dt.tz_localize(display_tz)
    if "volume" not in d.columns:
        d["volume"] = 0
    return d[["time", "value", "volume"]]

# ==========================
# Axis helpers
# ==========================

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

def session_frame(base_ts_jst, session_tz, display_tz, start_hm, end_hm):
    """基準時刻 day のセッション枠（表示tzへ変換）"""
    b = base_ts_jst.tz_convert(session_tz)
    d = b.date()
    start = pd.Timestamp(d.year, d.month, d.day, start_hm[0], start_hm[1], tz=session_tz)
    end = pd.Timestamp(d.year, d.month, d.day, end_hm[0], end_hm[1], tz=session_tz)
    return start.tz_convert(display_tz), end.tz_convert(display_tz)

# ==========================
# Plot
# ==========================

def plot_df(df, index_key, label, mode, tz, frame=None):
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.grid(True, alpha=0.3)

    if df.empty:
        # 何もない時も枠/軸は出す
        ax1.text(0.5, 0.5, "No data", transform=ax1.transAxes, ha="center", va="center", color="#9aa3bd", fontsize=22)
        ax1.set_title(f"{norm_key(index_key).upper()} ({label})", color="#ffb6c1")
        ax1.set_xlabel("Time" if mode == "1d" else "Date")
        ax1.set_ylabel("Index Value")
        if frame is not None:
            ax1.set_xlim(frame)
        format_time_axis(ax1, mode, tz)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{norm_key(index_key)}_{label}.png", dpi=180)
        plt.close()
        log(f"saved empty: {index_key}_{label}")
        return

    # 1d 色分け
    if mode == "1d":
        op, cl = df["value"].iloc[0], df["value"].iloc[-1]
        color = COLOR_UP if cl > op else (COLOR_DOWN if cl < op else COLOR_EQUAL)
        lw = 2.2
    else:
        color = COLOR_PRICE_DEFAULT
        lw = 1.8

    # volume
    if "volume" in df.columns and pd.notna(df["volume"]).sum() > 0 and df["volume"].abs().sum() > 0:
        ax2 = ax1.twinx()
        ax2.bar(df["time"], df["volume"], width=0.9 if mode == "1d" else 0.8, color=COLOR_VOLUME, alpha=0.35, zorder=1, label="Volume")

    ax1.plot(df["time"], df["value"], color=color, lw=lw, solid_capstyle="round", label="Index", zorder=3)
    ax1.set_title(f"{norm_key(index_key).upper()} ({label})", color="#ffb6c1")
    ax1.set_xlabel("Time" if mode == "1d" else "Date")
    ax1.set_ylabel("Index Value")

    format_time_axis(ax1, mode, tz)
    apply_y_padding(ax1, df["value"])
    if frame is not None:
        ax1.set_xlim(frame)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = f"{OUTPUT_DIR}/{norm_key(index_key)}_{label}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    log(f"saved: {outpath}")

# ==========================
# Main
# ==========================

def looks_like_return(series: pd.Series) -> bool:
    """
    値が差分/リターンっぽいか簡易判定:
      ・範囲が ±20 以内
      ・中央値が小さい（絶対値 < 15）
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    rng = s.max() - s.min()
    med = s.median()
    return (rng <= 20) and (abs(med) <= 15)

def restore_absolute_from_history(df_1d: pd.DataFrame, daily_hist: pd.DataFrame) -> pd.DataFrame:
    """1d が差分/リターンに見える場合、前日終値から絶対値へ復元"""
    if df_1d.empty:
        return df_1d
    if daily_hist.empty:
        return df_1d
    # 当日の 0:00 基準で直近営業日を探す
    day = df_1d["time"].iloc[-1].tz_convert(daily_hist["time"].iloc[-1].tzinfo).date()
    prev_close = daily_hist.loc[daily_hist["time"].dt.date < day, "value"]
    if prev_close.empty:
        return df_1d
    base = float(prev_close.iloc[-1])
    s = df_1d["value"].astype(float)
    # 値が ±0.5 〜 ±10% 程度なら % とみなす
    if s.abs().max() <= 50 and s.abs().median() <= 15:
        # -3.2 → base*(1 + -3.2/100)
        restored = base * (1.0 + s / 100.0)
    else:
        # 単純差分とみなして base + x
        restored = base + s
    out = df_1d.copy()
    out["value"] = restored
    return out

def main():
    index_key = os.environ.get("INDEX_KEY", "").strip()
    if not index_key:
        raise SystemExit("ERROR: INDEX_KEY not set")

    MP = market_profile(index_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    intraday_path = find_intraday(OUTPUT_DIR, index_key)
    history_path = find_history(OUTPUT_DIR, index_key)

    intraday = read_any(intraday_path, MP["RAW_TZ_INTRADAY"], MP["DISPLAY_TZ"], index_key) if intraday_path else pd.DataFrame()
    history  = read_any(history_path,  MP["RAW_TZ_HISTORY"],  MP["DISPLAY_TZ"], index_key) if history_path else pd.DataFrame()

    # history は日次想定
    daily_all = to_daily_from_history(history, MP["DISPLAY_TZ"]) if not history.empty else pd.DataFrame()

    # ---------- 1d ----------
    df_1d = pd.DataFrame()
    frame_1d = None

    if not intraday.empty:
        last_ts = intraday["time"].max()
        start_jst, end_jst = session_frame(last_ts, MP["SESSION_TZ"], MP["DISPLAY_TZ"], MP["SESSION_START"], MP["SESSION_END"])
        mask = (intraday["time"] >= start_jst) & (intraday["time"] <= end_jst)
        sliced = intraday.loc[mask].copy()

        if sliced.empty:
            # セッション厳格切り出しで消えた場合 → 当日全体でフォールバック
            day = last_ts.tz_convert(MP["DISPLAY_TZ"]).date()
            m2 = intraday["time"].dt.date == day
            sliced = intraday.loc[m2].copy()
            # それでも空なら最後の 6.5h だけ取る（日本 9:00-15:30 相当）
            if sliced.empty:
                sliced = intraday[intraday["time"] >= (last_ts - pd.Timedelta(hours=6, minutes=30))].copy()

        # 1d が差分/リターンなら前日終値から復元
        if looks_like_return(sliced["value"]) and not daily_all.empty:
            sliced = restore_absolute_from_history(sliced, daily_all)

        df_1d = sliced
        frame_1d = (start_jst, end_jst)

    plot_df(df_1d, index_key, "1d", "1d", MP["DISPLAY_TZ"], frame=frame_1d)

    # ---------- 7d / 1m / 1y ----------
    now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
    if not daily_all.empty:
        windows = [("7d", 7), ("1m", 31), ("1y", 365)]
        for label, days in windows:
            sub = daily_all[daily_all["time"] >= (now - timedelta(days=days))]
            plot_df(sub, index_key, label, "long", MP["DISPLAY_TZ"])
    else:
        # history が無い場合、1d から簡易日次生成（終値=最後、出来高=合計）
        if not intraday.empty:
            d = intraday.copy()
            d["date"] = d["time"].dt.tz_convert(MP["DISPLAY_TZ"]).dt.date
            g = d.groupby("date", as_index=False).agg({"value": "last", "volume": "sum"})
            g["time"] = pd.to_datetime(g["date"]).dt.tz_localize(MP["DISPLAY_TZ"])
            g = g[["time", "value", "volume"]].sort_values("time")
            now = pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
            for label, days in [("7d", 7), ("1m", 31), ("1y", 365)]:
                sub = g[g["time"] >= (now - timedelta(days=days))]
                plot_df(sub, index_key, label, "long", MP["DISPLAY_TZ"])
        else:
            # 何も無い場合でも雛形PNGを出力
            for label in ("7d", "1m", "1y"):
                plot_df(pd.DataFrame(), index_key, label, "long", MP["DISPLAY_TZ"])

if __name__ == "__main__":
    main()
