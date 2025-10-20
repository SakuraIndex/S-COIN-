#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

def load_intraday(csv_path: str, index_key: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if index_key not in df.columns:
        raise ValueError(f"CSVから '{index_key}' に対応する列を特定できません。候補: {set(df.columns)}")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime").sort_index()
    return df

def filter_session(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_t = datetime.strptime(start, "%H:%M").time()
    end_t = datetime.strptime(end, "%H:%M").time()
    return df.between_time(start_t, end_t)

def make_snapshot(df: pd.DataFrame, index_key: str, anchor_time: str, value_type: str):
    anchor_t = datetime.strptime(anchor_time, "%H:%M").time()
    anchor_value = df.between_time(anchor_t, anchor_t).iloc[0][index_key]
    last_value = df.iloc[-1][index_key]

    if value_type == "percent":
        change_pct = last_value - anchor_value
    else:
        change_pct = (last_value / anchor_value - 1.0) * 100

    return round(change_pct, 3)

def plot_intraday(df: pd.DataFrame, index_key: str, out_path: str, session_start: str, session_end: str):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[index_key], color="cyan", linewidth=2)
    ax.set_title(f"{index_key} Intraday Snapshot ({datetime.now(JST):%Y/%m/%d %H:%M})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Anchor (%)")
    ax.grid(True, color="gray", linestyle="--", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-key", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-text", required=True)
    parser.add_argument("--snapshot-png", required=True)
    parser.add_argument("--session-start", required=True)
    parser.add_argument("--session-end", required=True)
    parser.add_argument("--day-anchor", required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--value-type", choices=["raw", "percent"], default="raw")
    args = parser.parse_args()

    df = load_intraday(args.csv, args.index_key)
    session_df = filter_session(df, args.session_start, args.session_end)
    if session_df.empty:
        raise ValueError("セッション内データがありません。")

    change_pct = make_snapshot(session_df, args.index_key, args.day_anchor, args.value_type)
    now_jst = datetime.now(JST)

    # JSON出力
    out_json = {
        "index_key": args.index_key,
        "label": args.index_key,
        "pct_intraday": change_pct,
        "basis": args.basis,
        "session": {
            "start": args.session_start,
            "end": args.session_end,
            "anchor": args.day_anchor,
        },
        "updated_at": now_jst.isoformat(),
    }
    pd.Series(out_json).to_json(args.out_json, indent=2, force_ascii=False)

    # 投稿テキスト
    text = (
        f"▲ {args.index_key} 日中スナップショット ({now_jst:%Y/%m/%d %H:%M})\n"
        f"{change_pct:+.2f}%（基準: {args.basis}）\n"
        f"#{args.index_key} #日本株"
    )
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write(text)

    # チャート出力
    plot_intraday(session_df, args.index_key, args.snapshot_png, args.session_start, args.session_end)

if __name__ == "__main__":
    main()
