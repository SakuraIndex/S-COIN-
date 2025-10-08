#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d/7d/1m/1y) for Sakura Index series.
- 1d は市場セッションで切り出し（S-COIN+ は JST 9:00-15:30）
- Intraday が % の時は前日終値をアンカーにして絶対値へ変換
- Intraday 時刻が JST セッションから大きく外れていたら UTC とみなして再解釈
"""

import os, re
from datetime import timedelta
from typing import Union, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# ---------- slug 正規化 ----------
def norm_slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("-", "")
    s = s.replace("+", "_plus")
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s

# ---------- 市場プロファイル ----------
def market_profile(index_key: str) -> dict:
    k=(index_key or "").lower()
    # AIN-10 / Astra4: 米国株（ET 9:30-16:00 → JST 表示）
    if k in ("ain10","ain-10","astra4"):
        return dict(RAW_TZ_INTRADAY="America/New_York",
                    RAW_TZ_HISTORY="Asia/Tokyo",
                    DISPLAY_TZ="Asia/Tokyo",
                    SESSION_TZ="America/New_York",
                    SESSION_START=(9,30), SESSION_END=(16,0))
    # S-COIN+: 日本株（JST 9:00-15:30）
    if k in ("scoin+","scoin_plus","scoinplus","s-coin+"):
        return dict(RAW_TZ_INTRADAY="Asia/Tokyo",
                    RAW_TZ_HISTORY="Asia/Tokyo",
                    DISPLAY_TZ="Asia/Tokyo",
                    SESSION_TZ="Asia/Tokyo",
                    SESSION_START=(9,0), SESSION_END=(15,30))
    # R-BANK9: 日本株（JST 9:00-15:00）
    if k in ("rbank9","r-bank9","r_bank9"):
        return dict(RAW_TZ_INTRADAY="Asia/Tokyo",
                    RAW_TZ_HISTORY="Asia/Tokyo",
                    DISPLAY_TZ="Asia/Tokyo",
                    SESSION_TZ="Asia/Tokyo",
                    SESSION_START=(9,0), SESSION_END=(15,0))
    # fallback
    return dict(RAW_TZ_INTRADAY="Asia/Tokyo",
                RAW_TZ_HISTORY="Asia/Tokyo",
                DISPLAY_TZ="Asia/Tokyo",
                SESSION_TZ="Asia/Tokyo",
                SESSION_START=(9,0), SESSION_END=(15,0))

# ---------- 入出力 ----------
def _first(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p): return p
    return None

def alt_candidates(slug: str) -> List[str]:
    return list({slug, slug.replace("_",""), slug.replace("_plus","plus")})

def find_intraday(base, slug):
    c=[]
    for s in alt_candidates(slug):
        c += [f"{base}/{s}_intraday.csv", f"{base}/{s}_intraday.txt"]
    return _first(c)

def find_history(base, slug):
    c=[]
    for s in alt_candidates(slug):
        c += [
            f"{base}/{s}_history.csv",
            f"{base}/{s}_daily.csv",    # ワークフロー揺れ吸収
            f"{base}/{s}.csv"
        ]
    return _first(c)

# ---------- 解析ユーティリティ ----------
def parse_time_any(x, raw_tz: str, display_tz: str) -> Union[pd.Timestamp, pd.NaT]:
    if pd.isna(x): return pd.NaT
    s = str(x).strip()
    if re.fullmatch(r"\d{10}", s):
        return pd.Timestamp(int(s), unit="s", tz="UTC").tz_convert(display_tz)
    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t): return pd.NaT
    if t.tzinfo is None: t = t.tz_localize(raw_tz)
    return t.tz_convert(display_tz)

def pick_value_col(df: pd.DataFrame) -> str:
    cols=[c.lower() for c in df.columns]
    for k in ["close","price","value","index","終値"]:
        if k in cols: return df.columns[cols.index(k)]
    nums=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[0] if nums else df.columns[0]

def pick_volume_col(df: pd.DataFrame) -> Optional[str]:
    cols=[c.lower() for c in df.columns]
    for k in ["volume","vol","出来高"]:
        if k in cols: return df.columns[cols.index(k)]
    return None

def _reparse_as_utc(orig_series: pd.Series, display_tz: str) -> pd.Series:
    return orig_series.apply(lambda x: pd.Timestamp(x, tz="UTC").tz_convert(display_tz)
                                        if not pd.isna(x) else pd.NaT)

def read_any(path, raw_tz, display_tz) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["time","value","volume"])
    df=pd.read_csv(path)
    df.columns=[str(c).strip().lower() for c in df.columns]

    # 時刻列
    tcol=None
    for name in ["datetime","time","timestamp","date"]:
        if name in df.columns: tcol=name; break
    if not tcol:
        fuzzy=[c for c in df.columns if ("time" in c) or ("date" in c)]
        if fuzzy: tcol=fuzzy[0]
    if not tcol:
        raise KeyError(f"No time-like column found. columns={list(df.columns)}")

    vcol=pick_value_col(df); volcol=pick_volume_col(df)

    out=pd.DataFrame()
    # 1st pass: 指定 tz で解釈
    t1=df[tcol].apply(lambda x: parse_time_any(x, raw_tz, display_tz))
    out["time"]=t1
    out["value"]=pd.to_numeric(df[vcol], errors="coerce")
    out["volume"]=pd.to_numeric(df[volcol], errors="coerce") if volcol else 0
    out=out.dropna(subset=["time","value"]).sort_values("time").reset_index(drop=True)

    # JST セッションから大きく外れる場合は「元は UTC」とみなして再解釈
    if not out.empty and display_tz=="Asia/Tokyo":
        hrs=out["time"].dt.hour
        # 9-16 時が少なく、0-8 時が多いなら UTC と推定
        if (hrs.between(9,16).mean() < 0.2) and (hrs.between(0,8).mean() > 0.6):
            log("detected UTC-like timestamps → reparsing as UTC")
            t2=_reparse_as_utc(pd.to_datetime(df[tcol], errors="coerce"), display_tz)
            out["time"]=t2
            out=out.dropna(subset=["time","value"]).sort_values("time").reset_index(drop=True)

    log(f"read_any: rows={len(out)} path={path}  time_range={out['time'].min()}..{out['time'].max() if not out.empty else None}")
    return out

def to_daily(df: pd.DataFrame, display_tz: str) -> pd.DataFrame:
    if df.empty: return df
    d=df.copy()
    d["date"]=d["time"].dt.tz_convert(display_tz).dt.date
    g=d.groupby("date", as_index=False).agg({"value":"last","volume":"sum"})
    g["time"]=pd.to_datetime(g["date"]).dt.tz_localize(display_tz)
    return g[["time","value","volume"]]

# ---------- %→絶対値 変換 ----------
def _try_load_last_close(slug: str, daily_all: Optional[pd.DataFrame]) -> Optional[float]:
    v=os.environ.get("LAST_CLOSE","").strip()
    if v:
        try:
            x=float(v)
            if x>0: return x
        except: pass
    base_path=os.path.join(OUTPUT_DIR, f"{slug}_base.txt")
    if os.path.exists(base_path):
        try:
            with open(base_path,"r",encoding="utf-8") as f:
                x=float(f.read().strip())
            if x>0: return x
        except: pass
    if daily_all is not None and not daily_all.empty:
        try:
            x=float(pd.to_numeric(daily_all["value"], errors="coerce").dropna().iloc[-1])
            if x>50: return x
        except: pass
    return None

def normalize_intraday_scale(intraday: pd.DataFrame, daily_all: Optional[pd.DataFrame], slug: str) -> pd.DataFrame:
    if intraday.empty: return intraday
    s=pd.to_numeric(intraday["value"], errors="coerce")
    if s.isna().all(): return intraday
    max_abs=float(s.abs().max())
    is_decimal_pct = max_abs <= 0.2    # 0.0123 (=1.23%)
    is_pct         = max_abs <= 5.0    # 1.23 (%)
    if not (is_decimal_pct or is_pct): return intraday

    last_close=_try_load_last_close(slug, daily_all)
    if last_close is None:
        log("cannot find last_close (history or LAST_CLOSE missing) → keep % scale")
        return intraday

    pct = s * (100.0 if is_decimal_pct else 1.0)
    abs_val = last_close * (1.0 + pct/100.0)
    out=intraday.copy(); out["value"]=abs_val
    log(f"normalized %→abs using last_close={last_close}")
    return out

# ---------- 軸/描画 ----------
def format_time_axis(ax, mode, tz):
    if mode=="1d":
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    else:
        loc=mdates.AutoDateLocator(minticks=3, maxticks=6, tz=tz)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

def apply_y_padding(ax, series):
    s=pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: ax.set_ylim(0,1); return
    lo,hi=float(s.min()), float(s.max())
    pad=(hi-lo)*0.08 if hi!=lo else max(abs(lo)*0.02, 0.5)
    ax.set_ylim(lo-pad, hi+pad)

def session_frame_on_display(base_ts, session_tz, display_tz, start_hm, end_hm):
    base_in_sess = base_ts.tz_convert(session_tz)
    d=base_in_sess.date()
    start=pd.Timestamp(d.year,d.month,d.day,start_hm[0],start_hm[1], tz=session_tz)
    end  =pd.Timestamp(d.year,d.month,d.day,end_hm[0],end_hm[1], tz=session_tz)
    return start.tz_convert(display_tz), end.tz_convert(display_tz)

def session_frame_by_date(display_tz, session_tz, y,m,d, start_hm, end_hm):
    start=pd.Timestamp(y,m,d,start_hm[0],start_hm[1], tz=session_tz).tz_convert(display_tz)
    end  =pd.Timestamp(y,m,d,end_hm[0],end_hm[1], tz=session_tz).tz_convert(display_tz)
    return start, end

def today_session_frame(display_tz, session_tz, start_hm, end_hm):
    now = pd.Timestamp.now(tz=display_tz).tz_convert(session_tz)
    return session_frame_by_date(display_tz, session_tz, now.year, now.month, now.day, start_hm, end_hm)

def previous_session_frame(display_tz, session_tz, start_hm, end_hm):
    now = pd.Timestamp.now(tz=display_tz).tz_convert(session_tz) - pd.Timedelta(days=1)
    return session_frame_by_date(display_tz, session_tz, now.year, now.month, now.day, start_hm, end_hm)

def ensure_saved(fig, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout(); plt.savefig(outpath, dpi=180); plt.close(fig)
    log(f"saved: {outpath}")

def plot_df(df, slug, label, mode, tz, frame=None, placeholder_frame=None):
    outpath=f"{OUTPUT_DIR}/{slug}_{label}.png"

    if df.empty:
        fig, ax=plt.subplots(figsize=(9.5,4.8))
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{slug.upper()} ({label})", color="#ffb6c1")
        ax.set_xlabel("Time" if mode=="1d" else "Date"); ax.set_ylabel("Index Value")
        if placeholder_frame: ax.set_xlim(placeholder_frame); format_time_axis(ax, mode, tz)
        ax.text(0.5,0.5,"No data", color="#b8c2e0", ha="center", va="center", transform=ax.transAxes, fontsize=16)
        ensure_saved(fig, outpath); return

    if mode=="1d":
        open_p, close_p = df["value"].iloc[0], df["value"].iloc[-1]
        color = COLOR_UP if close_p>open_p else (COLOR_DOWN if close_p<open_p else COLOR_EQUAL)
        lw=2.2
    else:
        color, lw = COLOR_PRICE_DEFAULT, 1.8

    fig, ax1=plt.subplots(figsize=(9.5,4.8))
    ax1.grid(True, alpha=0.3)

    if "volume" in df and pd.to_numeric(df["volume"], errors="coerce").abs().sum()>0:
        ax2=ax1.twinx()
        ax2.bar(df["time"], df["volume"], width=0.9 if mode=="1d" else 0.8,
                color=COLOR_VOLUME, alpha=0.35, zorder=1)

    ax1.plot(df["time"], df["value"], color=color, lw=lw, solid_capstyle="round", zorder=3)
    ax1.set_title(f"{slug.upper()} ({label})", color="#ffb6c1")
    ax1.set_xlabel("Time" if mode=="1d" else "Date"); ax1.set_ylabel("Index Value")

    format_time_axis(ax1, mode, tz)
    apply_y_padding(ax1, df["value"])
    if frame: ax1.set_xlim(frame)

    ensure_saved(fig, outpath)

# ---------- 1d 窓（フォールバック付） ----------
def best_1d_window(intraday: pd.DataFrame, MP: dict) -> Tuple[pd.DataFrame, Optional[Tuple[pd.Timestamp,pd.Timestamp]], Tuple[pd.Timestamp,pd.Timestamp]]:
    disp_tz=MP["DISPLAY_TZ"]; sess_tz=MP["SESSION_TZ"]
    start_today, end_today = today_session_frame(disp_tz, sess_tz, MP["SESSION_START"], MP["SESSION_END"])
    placeholder=(start_today, end_today)

    if intraday.empty:
        return pd.DataFrame(), None, placeholder

    candidates=[]
    last_ts = intraday["time"].max()
    start1, end1 = session_frame_on_display(last_ts, sess_tz, disp_tz, MP["SESSION_START"], MP["SESSION_END"])
    candidates.append(("last-day", start1, end1))
    candidates.append(("today", start_today, end_today))
    start_prev, end_prev = previous_session_frame(disp_tz, sess_tz, MP["SESSION_START"], MP["SESSION_END"])
    candidates.append(("prev", start_prev, end_prev))

    for tag, a, b in candidates:
        sub = intraday[(intraday["time"]>=a) & (intraday["time"]<=b)]
        log(f"1d candidate={tag} rows={len(sub)} window=[{a}..{b}]")
        if len(sub)>=5:
            return sub.copy(), (a,b), placeholder

    # 直近 N 本（時間枠は “その日のセッション枠” を表示して見た目を合わせる）
    N=200
    tail=intraday.tail(N).copy()
    if not tail.empty:
        log(f"1d fallback tail rows={len(tail)} frame=placeholder({placeholder[0]}..{placeholder[1]})")
        return tail, placeholder, placeholder

    return pd.DataFrame(), None, placeholder

# ---------- メイン ----------
def main():
    index_key=os.environ.get("INDEX_KEY","").strip()
    if not index_key: raise SystemExit("ERROR: INDEX_KEY not set")
    slug=norm_slug(index_key)
    MP=market_profile(index_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    intraday_path=find_intraday(OUTPUT_DIR, slug)
    history_path =find_history (OUTPUT_DIR, slug)
    log(f"paths: intraday={intraday_path} history={history_path} slug={slug}")

    intraday=read_any(intraday_path, MP["RAW_TZ_INTRADAY"], MP["DISPLAY_TZ"]) if intraday_path else pd.DataFrame()
    history =read_any(history_path , MP["RAW_TZ_HISTORY"] , MP["DISPLAY_TZ"]) if history_path  else pd.DataFrame()
    daily_all=to_daily(history if not history.empty else intraday, MP["DISPLAY_TZ"])

    intraday=normalize_intraday_scale(intraday, daily_all, slug)

    # 1d
    df_1d, frame_1d, placeholder = best_1d_window(intraday, MP)
    plot_df(df_1d, slug, "1d", "1d", MP["DISPLAY_TZ"], frame=frame_1d, placeholder_frame=placeholder)

    # 7d/1m/1y
    now=pd.Timestamp.now(tz=MP["DISPLAY_TZ"])
    for label, days in [("7d",7),("1m",31),("1y",365)]:
        sub=daily_all[daily_all["time"] >= (now - timedelta(days=days))]
        log(f"{label} rows={len(sub)} (since {now - timedelta(days=days)})")
        plot_df(sub, slug, label, "long", MP["DISPLAY_TZ"])

if __name__ == "__main__":
    main()
