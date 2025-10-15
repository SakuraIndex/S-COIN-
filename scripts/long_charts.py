#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S-COIN+ charts + stats  (pct scale, dark theme)
"""
from pathlib import Path
import json
from datetime import datetime, timezone
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# constants / paths
# ------------------------
INDEX_KEY = "scoin_plus"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# ------------------------
# plotting style (dark)
# ------------------------
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
RED     = "#ff6b6b"

def _apply(ax, title: str) -> None:
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, alpha=0.6, linewidth=0.8)
    ax.tick_params(colors=FG_TEXT, labelsize=10)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel("Index / Value", color=FG_TEXT, fontsize=10)

def _save(df: pd.DataFrame, col: str, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots()
    _apply(ax, title)
    ax.plot(df.index, df[col], color=RED, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# data loading helpers
# ------------------------
def _pick_index_column(df: pd.DataFrame) -> str:
    """
    優先順位で S-COIN+ 列を決定。無ければ最後の列を使う。
    """
    # 正規化して比較（記号・アンダースコアの差異も吸収）
    def norm(s: str) -> str:
        return (
            s.strip().lower()
             .replace("+", "plus")
             .replace("-", "")
             .replace("_", "")
        )

    targets = {
        "scoinplus", "scoin", "scoinplu",  # 包含のため複数
        norm(INDEX_KEY), norm(INDEX_KEY.upper()), "scoinplusindex",
        "scoin_plus", "s_coin_plus", "scoin_pls"
    }
    ncols = {c: norm(c) for c in df.columns}
    for c, nc in ncols.items():
        if nc in targets:
            return c
    # fallback: 最後の列
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """
    intraday があれば intraday 優先、無ければ history。
    先頭列を DatetimeIndex にして NA を落とす。
    """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("S-COIN+: neither intraday nor history csv found.")
    df = df.dropna(how="all")
    # 数値化できる列だけ残す
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df

# ------------------------
# chart generation
# ------------------------
def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    tail_1d = df.tail(1000)
    tail_7d = df.tail(7 * 1000)

    _save(tail_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)")
    _save(tail_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)")

# ------------------------
# stats (pct) + marker writers
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker() -> None:
    """
    仕様:
      - intraday の S-COIN+ 列は “百分率[%]” を直接保持（例: -0.57 は -0.57%）
      - サイト側も pct として読む（scale="pct"）
      - post_intraday.txt には "+/-xx.xx%" を出力
    """
    df = _load_df()
    col = _pick_index_column(df)

    pct = None
    if len(df.index) > 0:
        last = df[col].iloc[-1]
        if pd.notna(last):
            pct = float(last)

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct is None else round(pct, 6),
        "scale": "pct",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if pct is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: {pct:+.2f}%\n", encoding="utf-8")

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
