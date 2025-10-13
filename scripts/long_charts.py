# scripts/long_charts.py  — S-COIN+
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

# ========= TZ =========
JP = pytz.timezone("Asia/Tokyo")
NY = pytz.timezone("America/New_York")

# ========= Theme =========
BG, FG, TITLE = "#0E1117", "#E6E6E6", "#f2b6c6"
UP, DOWN, GRID_A = "#3bd6c6", "#ff6b6b", 0.25  # up=青緑, down=赤

matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": FG,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": FG, "savefig.facecolor": BG,
})

OUTDIR = os.path.join("docs", "outputs")
os.makedirs(OUTDIR, exist_ok=True)

# ========== small utils ==========
def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _find_time_col(cols) -> str | None:
    cols = list(cols)
    if len(cols) == 0:
        return None
    for c in cols:
        if re.search(r"(time|日時|date|datetime|timestamp|時刻)", str(c), re.I):
            return c
    # 先頭列 fallback
    return cols[0]

def _to_datetime_jst(ser: pd.Series) -> pd.Series:
    t = pd.to_datetime(ser, errors="coerce", utc=True)
    if t.dt.tz is None:
        t = pd.to_datetime(ser, errors="coerce").dt.tz_localize("UTC")
    return t.dt.tz_convert(JP)

# ========== IO ==========
def read_intraday(path: str) -> pd.DataFrame:
    """
    intraday CSV → DataFrame(time[JST tz-aware], value[float])
    S-COIN+ は列構成が単純（time + value想定）だが、頑健に吸収。
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value"])
    raw = pd.read_csv(path, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=["time", "value"])
    df = _lower(raw)
    tcol = _find_time_col(df.columns)
    if tcol is None:
        return pd.DataFrame(columns=["time", "value"])

    # 値列推定
    vcol = None
    for c in df.columns:
        if c == tcol: continue
        if re.search(r"(value|index|mean|score|pct|change)", c, re.I):
            vcol = c; break
    if vcol is None:
        # 2列しかないなら先頭以外を値扱い
        if len(df.columns) >= 2:
            vcol = [c for c in df.columns if c != tcol][0]
        else:
            return pd.DataFrame(columns=["time", "value"])

    out = pd.DataFrame({
        "time": _to_datetime_jst(df[tcol]),
        "value": pd.to_numeric(df[vcol], errors="coerce")
    })
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def read_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "value"])
    h = pd.read_csv(path, dtype=str)
    if h.empty: return pd.DataFrame(columns=["date", "value"])
    h = _lower(h)
    dcol = None
    for c in h.columns:
        if re.search(r"(date|day|日時|time)", c, re.I):
            dcol = c; break
    if dcol is None: dcol = h.columns[0]
    vcol = None
    for c in h.columns:
        if c == dcol: continue
        if re.search(r"(value|index|mean|score|pct|change)", c, re.I):
            vcol = c; break
    if vcol is None and len(h.columns) > 1:
        vcol = h.columns[1]
    elif vcol is None:
        return pd.DataFrame(columns=["date", "value"])

    out = pd.DataFrame({
        "date": pd.to_datetime(h[dcol], errors="coerce"),
        "value": pd.to_numeric(h[vcol], errors="coerce")
    }).dropna().sort_values("date").reset_index(drop=True)
    return out

# ========== calc / window ==========
def pick_rolling24(df: pd.DataFrame) -> pd.DataFrame:
    """直近24時間（JST基準）"""
    if df.empty: return df
    now = pd.Timestamp.now(tz=JP)
    frm = now - pd.Timedelta(hours=24)
    w = df[(df["time"] >= frm) & (df["time"] <= now)]
    return w.reset_index(drop=True)

def resample_1min(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    g = df.set_index("time").sort_index()[["value"]].resample("1min").mean()
    g["value"] = g["value"].interpolate(limit_direction="both")
    return g.reset_index()

CAPS = {"astra4": 30.0, "rbank9": 30.0, "ain10": 30.0, "scoin_plus": 30.0}

def _clip(p: float, cap: float) -> float:
    if p is None or not pd.notna(p):
        return None
    return max(-cap, min(cap, float(p)))

def calc_percent(series: pd.Series, key: str) -> float | None:
    """
    S-COIN+ 特例:
      - すでに%系列（±数％レンジ）なら “そのままの最新値” を採用（-30%固定の元凶を回避）
    それ以外は、レベル/ポイント双方を試して小さい方を採用。
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return None
    first, last = float(s.iloc[0]), float(s.iloc[-1])

    if key == "scoin_plus":
        mean_abs = float(s.abs().mean())
        # “すでに%” らしきレンジ（0.5%〜5% 程度）を閾値に採用
        if 0.5 <= mean_abs <= 5.0:
            return _clip(last, CAPS.get(key, 30.0))

    # 1) level想定: 比率
    pct_ratio = ((last - first) / (abs(first) if abs(first) > 1e-9 else 1.0)) * 100.0
    # 2) %ポイント想定: 単純差
    pct_points = (last - first) * 100.0
    use = pct_points if abs(pct_points) < abs(pct_ratio) else pct_ratio
    return _clip(use, CAPS.get(key, 30.0))

# ========== plot ==========
def decorate(ax, title: str, xl: str, yl: str):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=GRID_A)
    for s in ax.spines.values(): s.set_color(FG)

def save(fig, path: str):
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

# ========== main ==========
def main():
    key = os.environ.get("INDEX_KEY", "scoin_plus").strip().lower()
    name = key.upper()
    intraday_csv = os.path.join(OUTDIR, f"{key}_intraday.csv")
    history_csv  = os.path.join(OUTDIR, f"{key}_history.csv")

    # intraday → 直近24h → 1分
    try:
        i = read_intraday(intraday_csv)
        i = pick_rolling24(i)
        i = resample_1min(i)
    except Exception as e:
        print("intraday load fail:", e)
        i = pd.DataFrame(columns=["time", "value"])

    # 騰落率
    delta = calc_percent(i["value"], key) if not i.empty else None
    color = UP if (delta is not None and delta >= 0) else DOWN

    # 1d（実質 24h）
    fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
    decorate(ax, f"{name} (1d)", "Time", "Index / %")
    if not i.empty:
        ax.plot(i["time"], i["value"], lw=2.2, color=color)
    else:
        ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
    save(fig, os.path.join(OUTDIR, f"{key}_1d.png"))

    # 7d/1m/1y（ヒストリがあれば）
    h = read_history(history_csv)
    for days, label in [(7,"7d"),(30,"1m"),(365,"1y")]:
        fig, ax = plt.subplots(figsize=(16,7), layout="constrained")
        decorate(ax, f"{name} ({label})", "Date", "Index / %")
        if not h.empty:
            hh = h.tail(days)
            if len(hh) >= 2:
                col = UP if hh["value"].iloc[-1] >= hh["value"].iloc[0] else DOWN
                ax.plot(hh["date"], hh["value"], lw=2.0, color=col)
            else:
                ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.5)
        else:
            ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center", alpha=0.5)
        save(fig, os.path.join(OUTDIR, f"{key}_{label}.png"))

    # %テキスト（サイトで使用）
    txt = f"{name} 1d: {(0.0 if delta is None else delta):+0.2f}%"
    with open(os.path.join(OUTDIR, f"{key}_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

if __name__ == "__main__":
    main()
