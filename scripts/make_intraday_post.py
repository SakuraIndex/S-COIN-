#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_intraday_post.py

Intraday CSV を読み取り、投稿テキスト (.txt)・統計 (.json)・スナップショット画像 (.png) を生成。
日本株（JST 09:00–15:30）を基本としつつ、24h銘柄にも対応。
旧CLIオプション (--basis/--day-anchor/--session-start/--session-end) も受け付ける。
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

JST = "Asia/Tokyo"

# ----------------------------- util -----------------------------

def _find_ts_and_value_columns(df: pd.DataFrame, index_key: str | None) -> tuple[str, str]:
    lowered = {c.lower(): c for c in df.columns}
    for cand in ["ts", "timestamp", "time", "datetime", "date", "time_jst", "datetime"]:
        if cand in lowered:
            ts_col = lowered[cand]
            break
    else:
        # 先頭列を時刻として試みる（最後の保険）
        ts_col = df.columns[0]

    value_col = None
    if index_key:
        for c in df.columns:
            lc = c.lower()
            if lc == index_key.lower() or lc.replace("+", "") == index_key.lower():
                value_col = c
                break

    if value_col is None and len(df.columns) == 2:
        value_col = [c for c in df.columns if c != ts_col][0]

    if value_col is None:
        for c in df.columns:
            if c == ts_col:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                value_col = c
                break

    if value_col is None:
        raise ValueError("値列が特定できません (index_key または数値列を確認)")

    return ts_col, value_col


def load_intraday(csv_path: str, index_key: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    ts_col, value_col = _find_ts_and_value_columns(df, index_key)

    ts = pd.to_datetime(df[ts_col], errors="coerce")
    # タイムゾーン整備（UTC/JST混在でもOKに）
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(JST, nonexistent="shift_forward", ambiguous="NaT")
    else:
        ts = ts.dt.tz_convert(JST)

    out = pd.DataFrame({
        "ts": ts,
        "value": pd.to_numeric(df[value_col], errors="coerce")
    }).dropna(subset=["ts"])
    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return out


def filter_session(df: pd.DataFrame, session: str | None) -> pd.DataFrame:
    if df.empty or not session or session.lower() in ["all", "24h"]:
        return df
    if "-" not in session:
        raise ValueError(f"セッション形式が不正です: {session!r}（例: 09:00-15:30）")

    start, end = session.split("-")
    sh, sm = [int(x) for x in start.split(":")]
    eh, em = [int(x) for x in end.split(":")]

    # 基準日は最新データのJST日付
    day = df["ts"].iloc[-1].tz_convert(JST)
    st = pd.Timestamp(day.year, day.month, day.day, sh, sm, tz=JST)
    en = pd.Timestamp(day.year, day.month, day.day, eh, em, tz=JST)
    return df[(df["ts"] >= st) & (df["ts"] <= en)].copy()


def calc_change_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"open": np.nan, "close": np.nan, "change_pct": np.nan}
    o = df["value"].iloc[0]
    c = df["value"].iloc[-1]
    pct = (c / o - 1.0) * 100.0 if o != 0 else np.nan
    return {"open": round(o, 4), "close": round(c, 4), "change_pct": round(pct, 2)}


def plot_snapshot(df: pd.DataFrame, label: str, out_png: str):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    if df.empty:
        ax.text(0.5, 0.5, "No intraday data", ha="center", va="center", fontsize=14)
    else:
        pct = (df["value"] / df["value"].iloc[0] - 1.0) * 100.0
        ax.plot(df["ts"].dt.tz_localize(None), pct, color="cyan", linewidth=2, label=label)

    now_str = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(JST).strftime("%Y/%m/%d %H:%M")
    ax.set_title(f"{label} Intraday Snapshot ({now_str})", fontsize=14, pad=10)
    ax.set_xlabel("Time (JST)")
    ax.set_ylabel("Change vs Open (%)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def generate_post_text(label: str, stats: dict, out_text: str):
    now_jst = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(JST).strftime("%Y/%m/%d %H:%M")
    sym = "▲" if stats["change_pct"] > 0 else "▼" if stats["change_pct"] < 0 else "■"
    change_str = f"{stats['change_pct']:+.2f}%" if not np.isnan(stats["change_pct"]) else "N/A"
    lines = [
        f"{sym} {label} 日中取引 ({now_jst})",
        f"{change_str}（前日終値比）",
        "※ 構成銘柄の等ウェイト平均",
        f"#{label.replace(' ', '')} #日本株",
    ]
    with open(out_text, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate intraday post/snapshot from CSV")
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--snapshot-png", default=None)
    ap.add_argument("--session", default=None, help='例: "09:00-15:30", "24h", "all"')
    ap.add_argument("--label", default=None)

    # 旧オプション（互換用・受け取って内部で解釈）
    ap.add_argument("--basis", default=None)            # 例: open@09:00 / stable@10:00（本スクリプトでは基準は「セッション先頭」固定・無視）
    ap.add_argument("--day-anchor", default=None)       # 例: JST@09:00（無視）
    ap.add_argument("--session-start", default=None)    # 例: 09:00
    ap.add_argument("--session-end", default=None)      # 例: 15:30

    args = ap.parse_args()

    # 互換：--session-start/--session-end が来たら --session を組み立て
    session = args.session
    if not session and (args.session_start or args.session_end):
        if not (args.session_start and args.session_end):
            raise SystemExit("Both --session-start and --session-end are required when one is provided.")
        session = f"{args.session_start}-{args.session_end}"

    # デフォルト：日本株の取引時間
    if session is None:
        session = "09:00-15:30"

    label = args.label or args.index_key.upper()

    df = load_intraday(args.csv, index_key=args.index_key)
    df = filter_session(df, session)
    stats = calc_change_stats(df)

    # JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({
            "index_key": args.index_key,
            "open": stats["open"],
            "close": stats["close"],
            "pct_1d": stats["change_pct"],
            "scale": "percent",
            "basis": f"session_open({session})",
            "updated_at": pd.Timestamp.utcnow().isoformat()
        }, f, ensure_ascii=False, indent=2)

    # TXT
    generate_post_text(label, stats, args.out_text)

    # PNG
    if args.snapshot_png:
        plot_snapshot(df, label, args.snapshot_png)

    print(f"Done: {args.index_key} change={stats['change_pct']}% session={session}")

if __name__ == "__main__":
    main()
