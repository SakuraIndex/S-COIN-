#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from datetime import datetime, time, timedelta, timezone

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

JST = timezone(timedelta(hours=9), name="JST")

def parse_args():
    p = argparse.ArgumentParser(description="Make intraday post text, json and snapshot image.")
    p.add_argument("--index-key", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-text", required=True)
    p.add_argument("--snapshot-png", required=True)
    p.add_argument("--session-start", required=True, help="HH:MM JST")
    p.add_argument("--session-end", required=True, help="HH:MM JST")
    p.add_argument("--day-anchor", required=True, help="HH:MM JST (labeling anchor)")
    p.add_argument("--basis", required=True, help='e.g. "open@09:00"')
    p.add_argument(
        "--value-type",
        choices=["ratio", "percent", "level"],
        default="ratio",
        help=(
            "ratio  : 値が水準のとき (pct = (last/anchor-1)*100)\n"
            "percent: 値がすでに%のとき (pct = last - anchor)\n"
            "level  : 値が水準だが差分[%pt]でなくポイント差を表示したい場合など拡張用"
        ),
    )
    p.add_argument("--label", default=None)
    return p.parse_args()

def _to_jst(ts):
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    return ts.dt.tz_convert(JST)

def load_intraday(csv_path: str, index_key: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Datetime 列名は固定
    if "Datetime" not in df.columns:
        raise ValueError("CSVに 'Datetime' 列がありません。")
    # 値の列名（大小文字・ハイフン/アンダースコアゆるく吸収）
    possible_cols = [
        index_key,
        index_key.upper(),
        index_key.lower(),
        index_key.replace("-", "_"),
        index_key.replace("_", "-"),
        index_key.replace("+", "_"),
        index_key.replace("+", "-"),
        index_key.replace("-", ""),
        index_key.replace("_", ""),
    ]
    val_col = None
    for c in df.columns:
        if c in possible_cols:
            val_col = c
            break
    if val_col is None:
        # 代表例に S-COIN+ と S_COIN_PLUS など
        alt = index_key.replace("-", "_").replace("+", "_")
        for c in df.columns:
            if c.lower() == alt.lower():
                val_col = c
                break
    if val_col is None:
        raise ValueError(f"CSVから '{index_key}' に対応する列名を特定できません。候補: {set(df.columns)}")

    # 時刻をUTCとして読み → JSTへ
    ts = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    if ts.isna().all():
        # UTCマークがない生JSTの可能性 → tz_localize(JST)
        ts = pd.to_datetime(df["Datetime"], errors="coerce")
        ts = ts.dt.tz_localize(JST)

    df = pd.DataFrame({"ts": ts.dt.tz_convert(JST), "val": pd.to_numeric(df[val_col], errors="coerce")})
    df = df.dropna(subset=["ts", "val"]).sort_values("ts")
    return df.reset_index(drop=True)

def today_session_bounds(df: pd.DataFrame, start_hhmm: str, end_hhmm: str):
    if df.empty:
        raise ValueError("セッション内データがありません。")
    today = df["ts"].iloc[-1].date()  # 最新行の日付を採用
    sh, sm = map(int, start_hhmm.split(":"))
    eh, em = map(int, end_hhmm.split(":"))
    start = datetime(today.year, today.month, today.day, sh, sm, tzinfo=JST)
    end   = datetime(today.year, today.month, today.day, eh, em, tzinfo=JST)
    if end <= start:
        end = end + timedelta(days=1)
    return start, end

def slice_session(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    return df[(df["ts"] >= start) & (df["ts"] <= end)].copy()

def compute_change_percent(df_sess: pd.DataFrame, value_type: str) -> tuple[float, float]:
    """
    return (anchor_value, pct_change)
    - ratio   : pct = (last/anchor - 1) * 100
    - percent : pct = last - anchor           ← CSVが%のときはコレ
    - level   : 同上だがlevelをそのまま差分とみなす（拡張用）
    """
    if df_sess.empty:
        raise ValueError("セッション内データがありません。")
    anchor = float(df_sess["val"].iloc[0])
    last   = float(df_sess["val"].iloc[-1])

    if value_type == "percent":
        pct = last - anchor
    elif value_type == "level":
        pct = last - anchor
    else:  # ratio (デフォルト)
        if anchor == 0:
            pct = 0.0
        else:
            pct = (last / anchor - 1.0) * 100.0
    return anchor, pct

def format_signed(x: float, digits: int = 2) -> str:
    s = f"{abs(x):.{digits}f}"
    return f"+{s}" if x >= 0 else f"-{s}"

def plot_snapshot(df_sess: pd.DataFrame, anchor_ts: datetime, title: str, out_png: str):
    # 黒ベース＋枠線ゼロ
    plt.figure(figsize=(10, 5), facecolor="#0d0f12")
    ax = plt.gca()
    ax.set_facecolor("#0d0f12")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ライン
    ax.plot(df_sess["ts"], df_sess["val"], linewidth=2.0, color="#28d7da")

    # 目盛り薄灰
    ax.tick_params(colors="#c7ccd1")
    ax.xaxis.set_major_locator(MaxNLocator(8))
    ax.yaxis.set_major_locator(MaxNLocator(8))
    ax.grid(color="#262a31", alpha=0.6)

    ax.set_title(title, color="#e6ebf0")
    ax.set_xlabel("Time", color="#c7ccd1")
    ax.set_ylabel("Change vs Anchor (%)", color="#c7ccd1")

    plt.tight_layout()
    plt.savefig(out_png, dpi=140, facecolor="#0d0f12", bbox_inches="tight")
    plt.close()

def main():
    args = parse_args()

    df_all = load_intraday(args.csv, args.index_key)
    sess_start, sess_end = today_session_bounds(df_all, args.session_start, args.session_end)
    df_sess = slice_session(df_all, sess_start, sess_end)
    if df_sess.empty:
        raise ValueError("セッション内データがありません。")

    # “アンカーを 0% に合わせる” ため、値系列をアンカー起点に変換
    # percent の場合 → 値は%なので「差分で0起点」に
    # ratio/level の場合 → (val/anchor-1)*100 を系列に
    anchor_first = float(df_sess["val"].iloc[0])
    if args.value_type == "percent":
        df_plot = df_sess.copy()
        df_plot["val"] = df_plot["val"] - anchor_first
    elif args.value_type == "level":
        df_plot = df_sess.copy()
        df_plot["val"] = (df_plot["val"] - anchor_first)  # level差（必要なら×100等に拡張）
    else:
        df_plot = df_sess.copy()
        if anchor_first == 0:
            df_plot["val"] = 0.0
        else:
            df_plot["val"] = (df_plot["val"] / anchor_first - 1.0) * 100.0

    # 騰落率（テキスト/JSONに載せる最終値）
    _, pct_intraday = compute_change_percent(df_sess, args.value_type)

    # 画像
    title = f"{args.index_key.upper()} Intraday Snapshot ({datetime.now(JST).strftime('%Y/%m/%d %H:%M')})"
    plot_snapshot(df_plot, df_plot["ts"].iloc[0], title, args.snapshot_png)

    # テキスト
    post_lines = [
        f"▲ {args.index_key.replace('_',' ').upper()} 日中スナップショット ({datetime.now(JST).strftime('%Y/%m/%d %H:%M')})",
        f"{format_signed(pct_intraday)}%（基準: {args.basis}）",
        f"#{args.index_key.replace('_','+').upper()} #日本株",
    ]
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write("\n".join(post_lines) + "\n")

    # JSON
    payload = {
        "index_key": args.index_key,
        "label": args.index_key.upper(),
        "pct_intraday": pct_intraday,
        "basis": args.basis,
        "session": {
            "start": args.session_start,
            "end": args.session_end,
            "anchor": args.day_anchor,
        },
        "updated_at": datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
