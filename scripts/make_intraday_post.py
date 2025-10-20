#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_intraday_post.py

等加重インデックスの「日中スナップショット画像」と
「SNS投稿用テキスト」を生成するユーティリティ。

- CSV列名の自動解決に対応（例: index_key=scoin_plus -> 実列名 "S-COIN+" を解決）
- JSTセッション（開始/終了）でのフィルタ
- アンカー時刻（例: 09:00）を「open@09:00」等の基準値として採用
- スナップショットPNG（matplotlib）
- 投稿テキスト（.txt）
- JSON統計（任意）

CSV 前提:
    - ヘッダに "Datetime" 列（ISO, UTC基準 or ローカルでもOK）
    - インデックス列（例: "S-COIN+" / "R-BANK9" など）

使い方（例）:
    python scripts/make_intraday_post.py \
      --index-key scoin_plus \
      --csv docs/outputs/scoin_plus_intraday.csv \
      --out-text docs/outputs/scoin_plus_post_intraday.txt \
      --snapshot-png docs/outputs/scoin_plus_intraday.png \
      --session-start 09:00 \
      --session-end 15:30 \
      --day-anchor 09:00 \
      --basis open@09:00 \
      --label "S-COIN+"

"""

import argparse
import json
import math
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ===== ユーティリティ =====

JST = "Asia/Tokyo"


def norm_token(s: str) -> str:
    """英数字のみ小文字化に正規化。列名の近似一致に利用。"""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def resolve_column(index_key: str, columns) -> str:
    """
    index_key(論理名)からCSVの実列名を解決する。
    1) エイリアスマップ優先
    2) 正規化(英数字のみ小文字化)して近似一致
    """

    # よく使う別名をここに追加
    alias = {
        "scoin_plus": "S-COIN+",
        "rbank9": "R-BANK9",
        "ain10": "AIN-10",
    }

    if index_key in alias and alias[index_key] in columns:
        return alias[index_key]

    target = norm_token(index_key)
    for c in columns:
        if norm_token(c) == target:
            return c

    raise ValueError(
        f"CSVには '{index_key}' に対応する列が見つかりません。候補: {list(columns)}"
    )


def parse_hhmm(s: str) -> Tuple[int, int]:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s.strip())
    if not m:
        raise ValueError(f"時刻はHH:MM形式で指定してください: {s!r}")
    hh = int(m.group(1))
    mm = int(m.group(2))
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError(f"不正な時刻です: {s!r}")
    return hh, mm


def ensure_jst(ts: pd.Timestamp) -> pd.Timestamp:
    """
    与えられたTimestampをJSTに。
    naive   -> tz_localize("UTC").tz_convert(JST) とみなす
    tzあり  -> tz_convert(JST)
    """
    if ts.tzinfo is None:
        return ts.tz_localize("UTC").tz_convert(JST)
    return ts.tz_convert(JST)


def utcnow_jst() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(JST)


@dataclass
class Args:
    index_key: str
    csv: str
    out_text: str
    out_json: Optional[str]
    snapshot_png: Optional[str]
    label: Optional[str]
    session_start: str
    session_end: str
    day_anchor: str
    basis: str  # 文字ラベル（例：open@09:00）


# ===== データ読み込み・処理 =====

def load_intraday(csv_path: str, index_key: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Datetime" not in df.columns:
        raise ValueError("CSVに 'Datetime' 列が必要です。")

    col = resolve_column(index_key, df.columns)

    # Datetime -> pandas.Timestamp（JSTに揃える）
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime"]).copy()
    df["Datetime"] = df["Datetime"].dt.tz_convert(JST)

    # 値列を 'value' に統一
    df = df[["Datetime", col]].rename(columns={col: "value"})
    df = df.sort_values("Datetime")
    return df


def filter_session(df: pd.DataFrame, session_start: str, session_end: str) -> pd.DataFrame:
    """JSTの同一営業日で時刻帯だけを抽出（'HH:MM' 指定）。"""
    if df.empty:
        return df

    start_h, start_m = parse_hhmm(session_start)
    end_h, end_m = parse_hhmm(session_end)

    # 最新データのローカル日付を基準に、その日のHH:MM帯を抽出
    last_ts = df["Datetime"].iloc[-1]
    day = last_ts.tz_convert(JST).date()

    start = pd.Timestamp(year=day.year, month=day.month, day=day.day,
                         hour=start_h, minute=start_m, tz=JST)
    end = pd.Timestamp(year=day.year, month=day.month, day=day.day,
                       hour=end_h, minute=end_m, tz=JST)

    return df[(df["Datetime"] >= start) & (df["Datetime"] <= end)].copy()


def find_anchor_value(df: pd.DataFrame, anchor_hhmm: str) -> Tuple[pd.Timestamp, float]:
    """同一日の anchor時刻（最初の >= anchor）に最も近い値を返す。"""
    if df.empty:
        raise ValueError("セッション内データが空です。")

    hh, mm = parse_hhmm(anchor_hhmm)
    day = df["Datetime"].iloc[-1].tz_convert(JST).date()
    anchor_ts = pd.Timestamp(year=day.year, month=day.month, day=day.day,
                             hour=hh, minute=mm, tz=JST)
    # アンカー直後の最初のサンプル
    df_after = df[df["Datetime"] >= anchor_ts]
    if df_after.empty:
        # 直前のデータで代用
        df_before = df[df["Datetime"] < anchor_ts]
        if df_before.empty:
            raise ValueError("アンカー時刻近傍のデータが見つかりません。")
        row = df_before.iloc[-1]
    else:
        row = df_after.iloc[0]

    return row["Datetime"], float(row["value"])


def pct_change(cur: float, base: float) -> float:
    if base == 0 or not np.isfinite(base) or not np.isfinite(cur):
        return float("nan")
    return (cur / base - 1.0) * 100.0


# ===== 出力 =====

def make_post_text(label: str,
                   pct_intraday: float,
                   now_ts: pd.Timestamp,
                   basis_label: str,
                   tickers_hint: Optional[str] = None) -> str:
    """
    投稿用テキスト（4行程度）。必要に応じて調整してください。
    """
    arrow = "▲" if pct_intraday >= 0 else "▼"
    pct_str = f"{pct_intraday:+.2f}%"
    jst_str = now_ts.strftime("%Y/%m/%d %H:%M")

    lines = []
    lines.append(f"{arrow} {label} 日中スナップショット ({jst_str})")
    lines.append(f"{pct_str}（基準: {basis_label}）")
    if tickers_hint:
        lines.append(f"※ 構成銘柄：{tickers_hint}")
    lines.append(f"#{label.replace(' ', '_')} #日本株")
    return "\n".join(lines) + "\n"


def save_snapshot_png(df: pd.DataFrame,
                      png_path: str,
                      label: str,
                      basis_ts: pd.Timestamp,
                      anchor_label: str):
    """
    シンプルなラインチャートをPNGで出力。
    """
    if df.empty:
        raise ValueError("スナップショット用データが空です。")

    # Y軸は「前日終値比の%」ではなく、本スクリプトは「アンカー基準からの%」で描きます。
    # グラフ視認性のため、アンカー基準に正規化。
    _, base_v = find_anchor_value(df, anchor_label.split("@")[-1])
    y = (df["value"] / base_v - 1.0) * 100.0

    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    ax.plot(df["Datetime"].dt.tz_convert(JST), y, linewidth=2)
    ax.set_title(f"{label} Intraday Snapshot ({basis_ts.strftime('%Y/%m/%d %H:%M')})", pad=12)
    ax.set_ylabel("Change vs Anchor (%)")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.25)
    # 余白
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    plt.close()


# ===== メイン =====

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-key", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-text", required=True)
    p.add_argument("--out-json", default=None)
    p.add_argument("--snapshot-png", default=None)
    p.add_argument("--label", default=None)

    # セッション／アンカー／基準ラベル（すべてJSTベース）
    p.add_argument("--session-start", required=True, help="HH:MM JST")
    p.add_argument("--session-end", required=True, help="HH:MM JST")
    p.add_argument("--day-anchor", required=True, help="HH:MM JST")
    p.add_argument("--basis", required=True, help="例: open@09:00")

    args_ns = p.parse_args()
    args = Args(
        index_key=args_ns.index_key,
        csv=args_ns.csv,
        out_text=args_ns.out_text,
        out_json=args_ns.out_json,
        snapshot_png=args_ns.snapshot_png,
        label=args_ns.label,
        session_start=args_ns.session_start,
        session_end=args_ns.session_end,
        day_anchor=args_ns.day_anchor,
        basis=args_ns.basis,
    )

    # ラベル既定値
    label = args.label or args.index_key.upper()

    # 1) CSV読み込み
    df_all = load_intraday(args.csv, args.index_key)

    # 2) セッションで絞る
    df = filter_session(df_all, args.session_start, args.session_end)
    if df.empty:
        raise ValueError("指定セッション内にデータがありません。")

    # 3) アンカー値（例: 09:00 近傍）を取得
    anchor_ts, anchor_val = find_anchor_value(df, args.day_anchor)

    # 4) 最新値と%変化（アンカー基準）を算出
    now_row = df.iloc[-1]
    now_val = float(now_row["value"])
    now_ts = now_row["Datetime"].tz_convert(JST)
    pct_intraday = pct_change(now_val, anchor_val)

    # 5) 投稿テキストを保存
    post_text = make_post_text(
        label=label,
        pct_intraday=pct_intraday,
        now_ts=now_ts,
        basis_label=args.basis,
        tickers_hint=None,  # 必要なら構成銘柄一覧の短縮表示を渡す
    )
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write(post_text)

    # 6) スナップショットPNG（任意）
    if args.snapshot_png:
        save_snapshot_png(
            df=df,
            png_path=args.snapshot_png,
            label=label,
            basis_ts=now_ts,
            anchor_label=args.basis,  # "open@09:00" 等（中で 09:00 を抽出）
        )

    # 7) JSON統計（任意）
    if args.out_json:
        out = {
            "index_key": args.index_key,
            "label": label,
            "pct_intraday": pct_intraday,
            "basis": args.basis,
            "session": {
                "start": args.session_start,
                "end": args.session_end,
                "anchor": args.day_anchor,
            },
            "updated_at": now_ts.isoformat(),
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
