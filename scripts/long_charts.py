# scripts/long_charts.py
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d / 7d / 1m / 1y) for INDEX_KEY.

Inputs
- docs/outputs/<index>_intraday.csv  (columns: time,value[,volume] or wide-plate by tickers)
- docs/outputs/<index>_history.csv   (columns: date,value)

Outputs
- docs/outputs/<index>_1d.png
- docs/outputs/<index>_7d.png
- docs/outputs/<index>_1m.png
- docs/outputs/<index>_1y.png

ENV (optional)
- INDEX_KEY:       e.g. "astra4" / "scoin_plus" / "rbank9" / "ain10"
- DISPLAY_TZ:      target timezone (default: "Asia/Tokyo")
- SESSION_START:   intraday clamp start local time (default: "09:00")
- SESSION_END:     intraday clamp end   local time (default: "15:30")
"""

from __future__ import annotations
import os
from typing import List, Optional

import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

# --------------- Theme ---------------
BG     = "#0E1117"
FG     = "#E6E6E6"
TITLE  = "#f2b6c6"
ACCENT = "#3bd6c6"  # long horizon line
GRID_A = 0.25

UP_COLOR   = "#e74c3c"  # 🔺 up   = red
DOWN_COLOR = "#2ecc71"  # 🔻 down = green
FLAT_COLOR = "#9aa0a6"  # neutral/insufficient data

matplotlib.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": FG,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "text.color": FG,
    "grid.color": FG,
    "savefig.facecolor": BG,
})

# --------------- Defaults & Paths ---------------
OUTPUTS_DIR   = os.path.join("docs", "outputs")
INDEX_KEY     = os.environ.get("INDEX_KEY", "index").strip().lower()
INDEX_NAME    = INDEX_KEY.upper().replace("_", "")
DISPLAY_TZSTR = os.environ.get("DISPLAY_TZ", "Asia/Tokyo")
DISPLAY_TZ    = pytz.timezone(DISPLAY_TZSTR)

SESSION_START = os.environ.get("SESSION_START", "09:00")
SESSION_END   = os.environ.get("SESSION_END",   "15:30")

# --------------- Helpers ---------------
def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _pick_time_col(cols: List[str]) -> Optional[str]:
    for k in ("time", "timestamp", "date", "datetime"):
        if k in cols:
            return k
    for c in cols:
        if c.startswith("unnamed") and ": 0" in c:
            return c
    for c in cols:
        if ("time" in c) or ("date" in c):
            return c
    return None

def _decorate(ax, title: str, xl: str, yl: str):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(True, alpha=GRID_A)
    for sp in ax.spines.values():
        sp.set_color(FG)

def _save(fig, path: str):
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

# --------------- IO ---------------
def read_any_intraday(path: str) -> pd.DataFrame:
    """
    Return columns: time(tz-aware in DISPLAY_TZ), value, volume
    Accepts either long ("time,value[,volume]") or wide (tickers) formats.
    Robust to UTC/naive/tz-aware time stamps by normalizing to DISPLAY_TZ.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value", "volume"])

    raw = pd.read_csv(path, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = _lower(raw.copy())

    # drop comment columns like '# memo'
    drop = [c for c in df.columns if str(c).strip().startswith("#")]
    if drop:
        df = df.drop(columns=drop)

    tcol = _pick_time_col(df.columns.tolist())
    if tcol is None:
        raise KeyError(f"No time-like column in {path}")

    # detect value/volume for long format
    vcol, volcol = None, None
    for c in df.columns:
        lc = c
        if lc in ("value", "index", "score") or ("value" in lc):
            vcol = c
        if lc == "volume" or ("volume" in lc):
            volcol = c

    # --- normalize time to DISPLAY_TZ safely ---
    # 1) try parse without forcing utc to see if tz exists
    t = pd.to_datetime(df[tcol], errors="coerce", utc=False)

    if t.dt.tz is None:
        # naive → interpret as UTC then convert to DISPLAY_TZ
        t = pd.to_datetime(df[tcol], errors="coerce", utc=True)
        t = t.dt.tz_convert(DISPLAY_TZ)
    else:
        # already tz-aware → convert to DISPLAY_TZ directly
        t = t.dt.tz_convert(DISPLAY_TZ)

    out = pd.DataFrame({"time": t})

    if vcol is not None:
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
        out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if (volcol and volcol in df.columns) else 0
    else:
        # wide table → equally-weighted mean
        num_cols = [c for c in df.columns if c != tcol]
        vals = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        out["value"] = vals.mean(axis=1)
        out["volume"] = 0

    return out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)

def clamp_today_session_local(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only today's session in DISPLAY_TZ (default: 09:00–15:30)."""
    if df.empty:
        return df
    today = pd.Timestamp.now(tz=DISPLAY_TZ).normalize()
    start = pd.Timestamp(f"{today.date()} {SESSION_START}", tz=DISPLAY_TZ)
    end   = pd.Timestamp(f"{today.date()} {SESSION_END}",   tz=DISPLAY_TZ)
    m = (df["time"] >= start) & (df["time"] <= end)
    return df.loc[m].reset_index(drop=True)

def resample_minutes(df: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.set_index("time").sort_index()
    out = tmp[["value"]].resample(rule).mean()
    out["value"] = out["value"].interpolate(limit_direction="both")
    out["volume"] = 0
    return out.reset_index()

def read_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "value"])
    df = pd.read_csv(path)
    df = _lower(df)
    if "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

# --------------- Plotting ---------------
def plot_1d(i: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    _decorate(ax, f"{INDEX_NAME} (1d)", "Time", "Index Value")

    if i.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", alpha=0.6)
        _save(fig, out_path)
        return

    first = i["value"].iloc[0]
    last  = i["value"].iloc[-1]
    if pd.isna(first) or pd.isna(last):
        color = FLAT_COLOR
    else:
        diff = last - first
        color = UP_COLOR if diff > 0 else (DOWN_COLOR if diff < 0 else FLAT_COLOR)

    ax.plot(i["time"], i["value"], linewidth=2.6, color=color)
    _save(fig, out_path)

def plot_hist(h: pd.DataFrame, tail_n: int, label: str, out_path: str):
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    _decorate(ax, f"{INDEX_NAME} ({label})", "Date", "Index Value")
    hh = h.tail(tail_n)

    if len(hh) >= 2:
        ax.plot(hh["date"], hh["value"], linewidth=2.2, color=ACCENT)
    elif len(hh) == 1:
        ax.plot(hh["date"], hh["value"], marker="o", markersize=6, linewidth=0, color=ACCENT)
        y = hh["value"].iloc[0]
        ax.set_ylim(y - 0.1, y + 0.1)
        ax.text(0.5, 0.5, "Only 1 point (need ≥ 2)", transform=ax.transAxes,
                ha="center", va="center", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", alpha=0.5)

    _save(fig, out_path)

# --------------- Main ---------------
def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    intraday_csv = os.path.join(OUTPUTS_DIR, f"{INDEX_KEY}_intraday.csv")
    history_csv  = os.path.join(OUTPUTS_DIR, f"{INDEX_KEY}_history.csv")

    # ---- 1d ----
    try:
        i = read_any_intraday(intraday_csv)
        i = clamp_today_session_local(i)
        i = resample_minutes(i, "1min")
    except Exception as e:
        print(f"[WARN] intraday load failed: {e}")
        i = pd.DataFrame(columns=["time", "value", "volume"])

    plot_1d(i, os.path.join(OUTPUTS_DIR, f"{INDEX_KEY}_1d.png"))

    # ---- 7d / 1m / 1y ----
    h = read_history(history_csv)
    plot_hist(h, 7,   "7d", os.path.join(OUTPUTS_DIR, f"{INDEX_KEY}_7d.png"))
    plot_hist(h, 30,  "1m", os.path.join(OUTPUTS_DIR, f"{INDEX_KEY}_1m.png"))
    plot_hist(h, 365, "1y", os.path.join(OUTPUTS_DIR, f"{INDEX_KEY}_1y.png"))

    # Last-run memo
    with open(os.path.join(OUTPUTS_DIR, "_last_run.txt"), "w") as f:
        f.write(pd.Timestamp.now(tz=DISPLAY_TZ).isoformat())

if __name__ == "__main__":
    main()
