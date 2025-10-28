#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

INDEX_KEY = os.environ.get("INDEX_KEY", "scoin_plus")
OUT_DIR   = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

def apply_dark(fig, ax):
    fig.set_size_inches(16, 8)
    fig.set_dpi(160)
    fig.patch.set_facecolor("#111317")
    ax.set_facecolor("#111317")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(axis="both", colors="#ffffff", labelsize=10)
    ax.yaxis.label.set_color("#ffffff")
    ax.xaxis.label.set_color("#ffffff")
    ax.title.set_color("#ffffff")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.4, alpha=0.10, color="#ffffff")

def load_history() -> pd.DataFrame:
    f = OUT_DIR / f"{INDEX_KEY}_history.csv"
    if not f.exists(): return pd.DataFrame()
    df = pd.read_csv(f)
    if df.shape[1] < 2: return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df

def slice_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty: return df
    last = df["date"].max()
    return df[df["date"] >= (last - pd.Timedelta(days=days))].copy()

def plot(df: pd.DataFrame, title: str, out_png: Path):
    fig, ax = plt.subplots()
    apply_dark(fig, ax)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Level (index)", labelpad=10)
    ax.plot(df["date"].values, df["value"].values, linewidth=2.6)
    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[long] wrote: {out_png}")

def main():
    df = load_history()
    if df.empty:
        print("[long] history empty; nothing to draw."); return
    for tag, days in [("7d",7), ("1m",30), ("1y",365)]:
        d = slice_days(df, days)
        if d.empty: 
            print(f"[long] skip {tag}: empty"); 
            continue
        plot(d, f"{INDEX_KEY.upper()} ({tag} level)", OUT_DIR / f"{INDEX_KEY}_{tag}.png")

if __name__ == "__main__":
    main()
