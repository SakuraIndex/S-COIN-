#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S-COIN+ long-term charts (level)
  - Input: docs/outputs/<index>_history.csv (date,value)
  - Output: docs/outputs/<index>_{7d,1m,1y}.png
  - If historyが2点未満なら注意メッセージを1点だけ描く
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

INDEX_KEY = os.environ.get("INDEX_KEY", "scoin_plus")
OUT_DIR   = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

def diag(m): print(f"[long] {m}", flush=True)

def apply_dark(fig, ax):
    fig.set_size_inches(16, 8)
    fig.set_dpi(200)
    fig.patch.set_facecolor("#111317")
    ax.set_facecolor("#111317")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(axis="both", colors="#ffffff", labelsize=10)
    ax.yaxis.label.set_color("#ffffff")
    ax.xaxis.label.set_color("#ffffff")
    ax.title.set_color("#ffffff")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.10, alpha=0.10, color="#ffffff")

def load_history(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        diag(f"missing history: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2: return pd.DataFrame()
    # 列名吸収
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "value" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "value"})
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    # 重複除去
    df = df[~df["date"].duplicated(keep="last")]
    return df

def subset(df: pd.DataFrame, span: str) -> pd.DataFrame:
    if df.empty: return df
    days = {"7d":7, "1m":30, "1y":365}[span]
    last = df["date"].max()
    return df[df["date"] >= (last - pd.Timedelta(days=days))].copy()

def plot_level(df: pd.DataFrame, title: str, out_png: Path):
    fig, ax = plt.subplots()
    apply_dark(fig, ax)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Level (index)", labelpad=10)

    if len(df) >= 2:
        ax.plot(df["date"].values, df["value"].values, linewidth=2.6)
    elif len(df) == 1:
        ax.scatter(df["date"].values, df["value"].values, s=40)
        ax.text(0.03, 0.95, "Insufficient history (need ≥ 2 days)", transform=ax.transAxes, color="#9fb6c7")
    else:
        ax.text(0.03, 0.5, "No data", transform=ax.transAxes, color="#9fb6c7")

    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    diag(f"WROTE: {out_png}")

def main():
    hist = OUT_DIR / f"{INDEX_KEY}_history.csv"
    df = load_history(hist)
    if df.empty:
        diag("history empty; abort")
        return
    for span in ["7d","1m","1y"]:
        d = subset(df, span)
        out_png = OUT_DIR / f"{INDEX_KEY}_{span}.png"
        plot_level(d, f"{INDEX_KEY.upper()} ({span} level)", out_png)

if __name__ == "__main__":
    main()
