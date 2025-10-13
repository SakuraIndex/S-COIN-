# scripts/long_charts.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

# ---- TZ ----
JP = pytz.timezone("Asia/Tokyo")
NY = pytz.timezone("America/New_York")

# ---- Theme ----
BG, FG, TITLE = "#0E1117", "#E6E6E6", "#f2b6c6"
UP, DOWN, GRID_A = "#3bd6c6", "#ff6b6b", 0.25
matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": FG,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": FG, "savefig.facecolor": BG,
})

OUTDIR = os.path.join("docs", "outputs")
KEY = "scoin_plus"          # ← 固定
NAME = "SCOIN_PLUS"

def _lower(df): df.columns=[str(c).strip() for c in df.columns]; return df

def _find_time_col(cols):
    for c in cols:
        if re.search(r"time|日時|date|datetime|timestamp|時刻", str(c), re.I): return c
    return cols[0] if cols else None

def read_intraday(path:str)->pd.DataFrame:
    """docs/outputs/scoin_plus_intraday.csv を読み、JSTの tz-aware に整形"""
    if not os.path.exists(path): 
        return pd.DataFrame(columns=["time","value"])
    raw = pd.read_csv(path, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=["time","value"])
    df = _lower(raw)
    tcol = _find_time_col(df.columns)
    if tcol is None:
        return pd.DataFrame(columns=["time","value"])

    # 候補: "s-coin+" / "s_coin+" / "scoin_plus" / 最右列
    candidates = [c for c in df.columns if c.lower().replace(" ", "").replace("-", "").replace("_", "") in 
                  ("scoin+", "scoinplus", "scoin_plus", "s-coin+")]
    vcol = candidates[0] if candidates else df.columns[-1]

    # 時刻 → tz-aware(JST)
    t = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    if t.dt.tz is None:
        t = pd.to_datetime(df[tcol], errors="coerce").dt.tz_localize("UTC")
    t = t.dt.tz_convert(JP)

    out = pd.DataFrame({"time": t})
    out["value"] = pd.to_numeric(df[vcol], errors="coerce")
    out = out.dropna(subset=["time","value"]).sort_values("time").reset_index(drop=True)
    return out

def pick_window(df: pd.DataFrame) -> pd.DataFrame:
    """S-COIN+ は『直近24時間』をそのまま表示"""
    if df.empty: return df
    end = pd.Timestamp.now(tz=JP)
    beg = end - pd.Timedelta(hours=24)
    w = df[(df["time"]>=beg)&(df["time"]<=end)]
    return (w if not w.empty else df.tail(600)).reset_index(drop=True)

def calc_delta(series: pd.Series) -> float | None:
    """ヘッダ用の騰落率。S-COIN+ は正規化指数なので「差分×100（%ポイント近似）」が一番ブレが少ない。"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2: return None
    base, last = float(s.iloc[0]), float(s.iloc[-1])
    return (last - base) * 100.0

def decorate(ax, title, xl, yl):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=GRID_A)
    for sp in ax.spines.values(): sp.set_color(FG)

def save(fig, path):
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    intraday_csv = os.path.join(OUTDIR, f"{KEY}_intraday.csv")
    history_csv  = os.path.join(OUTDIR, f"{KEY}_history.csv")

    # --- load + window ---
    try:
        i = read_intraday(intraday_csv)
        i = pick_window(i)
        # 1分ごと補間（欠損が多いケースに備える）
        if not i.empty:
            i = (i.set_index("time")[["value"]]
                   .resample("1min").mean()
                   .interpolate(limit_direction="both")
                   .reset_index())
    except Exception as e:
        print("intraday load fail:", e)
        i = pd.DataFrame(columns=["time","value"])

    # --- delta & color ---
    delta = calc_delta(i["value"]) if not i.empty else None
    color = UP if (delta is not None and delta >= 0) else DOWN

    # --- 1d ---
    fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
    decorate(ax, f"{NAME} (1d)", "Time", "Index value")
    if not i.empty:
        ax.plot(i["time"], i["value"], lw=2.2, color=color)
    else:
        ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
    save(fig, os.path.join(OUTDIR, f"{KEY}_1d.png"))

    # --- long terms (7d/1m/1y は存在すれば描画) ---
    if os.path.exists(history_csv):
        h = pd.read_csv(history_csv)
        if "date" in h and "value" in h:
            h["date"]  = pd.to_datetime(h["date"], errors="coerce")
            h["value"] = pd.to_numeric(h["value"], errors="coerce")
            for days, label in [(7,"7d"),(30,"1m"),(365,"1y")]:
                fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
                decorate(ax, f"{NAME} ({label})", "Date", "Index value")
                hh = h.tail(days)
                if len(hh) >= 2:
                    col = UP if hh["value"].iloc[-1] >= hh["value"].iloc[0] else DOWN
                    ax.plot(hh["date"], hh["value"], lw=2.0, color=col)
                else:
                    ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.5)
                save(fig, os.path.join(OUTDIR, f"{KEY}_{label}.png"))

    # --- header text 出力 ---
    txt = f"S-COIN+ 1d: {(0.0 if delta is None else delta):+0.2f}%"
    with open(os.path.join(OUTDIR, f"{KEY}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

if __name__ == "__main__":
    main()
