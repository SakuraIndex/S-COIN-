#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_intraday_post.py

Usage Example (from GitHub Actions):
python scripts/make_intraday_post.py \
  --index-key scoin_plus \
  --csv docs/outputs/scoin_plus_intraday.csv \
  --out-json docs/outputs/scoin_plus_stats.json \
  --out-text docs/outputs/scoin_plus_post_intraday.txt \
  --snapshot-png docs/outputs/scoin_plus_intraday.png \
  --session-start "09:00" \
  --session-end   "15:30" \
  --day-anchor    "09:00" \
  --basis "open@09:00"
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Tuple, Optional, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== Theme (dark) =====
DARK_BG = "#0b0d10"   # figure 背景
AX_BG   = "#0b0d10"   # 軸背景
GRID    = "#2a2e35"
CYAN    = "#00e6e6"
TEXT    = "#e6edf3"


# ---------- Utilities ----------

def _norm_token(s: str) -> str:
    """アルファ数値のみを残して小文字化（照合用）"""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _find_datetime_col(cols: List[str]) -> str:
    """時刻カラム名を推定（先頭候補 / 'date' / 'time' を含む等）"""
    if not cols:
        raise ValueError("CSV にカラムが見つかりません。")
    # 1列目を最有力候補とみなす
    cand = cols[0]
    norm = _norm_token(cand)
    if any(k in norm for k in ["date", "time", "datetime", "timestamp"]):
        return cand
    # 他に 'date', 'time' を含むものがあればそれを優先
    for c in cols:
        n = _norm_token(c)
        if any(k in n for k in ["date", "time", "datetime", "timestamp"]):
            return c
    # 見当たらなければ先頭を採用
    return cols[0]


def _select_value_column(df: pd.DataFrame, index_key: str | None) -> str:
    """値カラムを決定。index_key があれば一致/準一致で選ぶ。
    候補が1つならそれを採用。複数あればエラーで候補表示。
    """
    cols = list(df.columns)
    dt_col = _find_datetime_col(cols)
    value_candidates = [c for c in cols if c != dt_col]
    if not value_candidates:
        raise ValueError("値カラムが見つかりません（時刻カラム以外が無い）。")

    if index_key:
        key_norm = _norm_token(index_key)
        # 1) 完全一致（正規化後）
        for c in value_candidates:
            if _norm_token(c) == key_norm:
                return c
        # 2) 部分一致（'scoin' など）
        for c in value_candidates:
            if key_norm in _norm_token(c):
                return c
        # 3) 'scoinplus' 対 'scoin' など、縮約で近い候補が1つだけなら採用
        close = [c for c in value_candidates if any(tok in _norm_token(c) for tok in key_norm.split())]
        if len(close) == 1:
            return close[0]
        # 見つからない → エラー
        msg = " | ".join(value_candidates)
        raise ValueError(f"CSV から '{index_key}' に対応する列を特定できません。候補: {msg}")

    # index_key 指定が無い場合
    if len(value_candidates) == 1:
        return value_candidates[0]
    msg = " | ".join(value_candidates)
    raise ValueError(f"値カラムを特定できません。--index-key を指定してください。候補: {msg}")


def load_intraday(csv_path: str, index_key: Optional[str]) -> pd.Series:
    """CSV を読み込み、(DatetimeIndex[JST], Series[float]) を返す。
    1列目を時刻列と解釈、対象列は index_key で選定（無ければ自動）。
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"CSV 読込に失敗: {csv_path} ({e})")

    if df.empty:
        raise ValueError("CSV が空です。")

    # 時刻列 & 値列の特定
    dt_col = _find_datetime_col(list(df.columns))
    val_col = _select_value_column(df, index_key)

    # 時刻をパース → JST に
    ts = pd.to_datetime(df[dt_col], errors="coerce", utc=False)
    if ts.dt.tz is None:
        # タイムゾーン未付与なら JST とみなす（ローカライズ）
        ts = ts.dt.tz_localize("Asia/Tokyo", nonexistent="shift_forward", ambiguous="NaT")
    else:
        # 付いているなら JST に変換
        ts = ts.dt.tz_convert("Asia/Tokyo")

    if ts.isna().all():
        raise ValueError("時刻列の解釈に失敗（すべて NaT）。CSV 時刻形式を確認してください。")

    s = pd.to_numeric(df[val_col], errors="coerce")
    ser = pd.Series(s.values, index=ts).dropna()
    ser = ser.sort_index()

    if ser.empty:
        raise ValueError("シリーズが空です（NaN のみ）。")

    return ser


def _parse_hhmm(hhmm: str) -> Tuple[int, int]:
    hh, mm = hhmm.strip().split(":")
    return int(hh), int(mm)


def filter_session(ser: pd.Series, session_start: str, session_end: str) -> pd.Series:
    """同一営業日の JST セッションでフィルタ。日付は末尾データの JST 日付を採用。"""
    if ser.index.tz is None:
        ser.index = ser.index.tz_localize("Asia/Tokyo")

    last_jst = ser.index[-1].tz_convert("Asia/Tokyo")
    day = last_jst.date()
    sh, sm = _parse_hhmm(session_start)
    eh, em = _parse_hhmm(session_end)

    start = pd.Timestamp(year=day.year, month=day.month, day=day.day,
                         hour=sh, minute=sm, tz="Asia/Tokyo")
    end = pd.Timestamp(year=day.year, month=day.month, day=day.day,
                       hour=eh, minute=em, tz="Asia/Tokyo")

    mask = (ser.index >= start) & (ser.index <= end)
    clipped = ser.loc[mask]
    # もしセッション内にデータが無ければ、当日分だけ抽出（保険）
    if clipped.empty:
        day_mask = (ser.index.date == day)
        clipped = ser.loc[day_mask]
    return clipped


def compute_anchor(series_in_session: pd.Series, day_anchor: str) -> Tuple[pd.Timestamp, float]:
    """アンカー時刻（例 09:00）以降の最初の値を基準にする。無ければセッション最初。"""
    if series_in_session.empty:
        raise ValueError("セッション内データが空です。")

    day = series_in_session.index[0].tz_convert("Asia/Tokyo").date()
    ah, am = _parse_hhmm(day_anchor)
    anchor_ts = pd.Timestamp(day.year, day.month, day.day, ah, am, tz="Asia/Tokyo")

    # アンカー以降の最初
    after = series_in_session.loc[series_in_session.index >= anchor_ts]
    if not after.empty:
        return after.index[0], float(after.iloc[0])

    # 取れなければセッション最初
    return series_in_session.index[0], float(series_in_session.iloc[0])


def to_change_percent(series: pd.Series, anchor_value: float) -> pd.Series:
    """(value / anchor - 1) * 100"""
    return (series / anchor_value - 1.0) * 100.0


def arrow_prefix(pct: float) -> str:
    if pct > 0:
        return "▲"
    if pct < 0:
        return "▼"
    return "■"


def human_label(index_key: str) -> str:
    """見出し用ラベル。渡し先に合わせて適宜調整。"""
    # 例: "scoin_plus" -> "SCOIN_PLUS"
    return index_key.upper()


# ---------- Plot ----------

def plot_snapshot(df_pct: pd.Series, out_png: str, label: str) -> None:
    """黒背景・白枠無しでスナップショット描画"""
    fig, ax = plt.subplots(figsize=(14, 7), dpi=120)

    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.25, color=GRID)
    ax.tick_params(colors=TEXT, labelsize=11)
    ax.set_xlabel("Time", color=TEXT, labelpad=6)
    ax.set_ylabel("Change vs Anchor (%)", color=TEXT, labelpad=6)

    ax.plot(df_pct.index, df_pct.values, linewidth=2.4, color=CYAN, label=label)

    ts = df_pct.index[-1].tz_convert("Asia/Tokyo").strftime("%Y/%m/%d %H:%M")
    ax.set_title(f"{label} Intraday Snapshot ({ts})", color=TEXT, fontsize=16, pad=10)

    plt.tight_layout(pad=0.3)
    fig.savefig(out_png, facecolor=DARK_BG, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-key", required=True, help="Series key (e.g., scoin_plus)")
    parser.add_argument("--csv", required=True, help="Input intraday CSV path")
    parser.add_argument("--out-json", required=True, help="Output stats JSON")
    parser.add_argument("--out-text", required=True, help="Output post text file")
    parser.add_argument("--snapshot-png", required=True, help="Output snapshot PNG")
    parser.add_argument("--session-start", required=True, help="HH:MM JST")
    parser.add_argument("--session-end", required=True, help="HH:MM JST")
    parser.add_argument("--day-anchor", required=True, help="HH:MM JST")
    parser.add_argument("--basis", required=True, help="Return basis label (e.g., open@09:00)")
    args = parser.parse_args()

    # 1) CSV 読込
    ser_all = load_intraday(args.csv, args.index_key)

    # 2) セッションで切り出し
    ser_sess = filter_session(ser_all, args.session_start, args.session_end)
    if ser_sess.empty:
        raise ValueError("指定セッション内にデータがありません。")

    # 3) アンカー算出
    anchor_ts, anchor_val = compute_anchor(ser_sess, args.day_anchor)

    # 4) 変化率(%) 化（アンカー基準）
    ser_pct = to_change_percent(ser_sess, anchor_val)
    last_pct = float(ser_pct.iloc[-1])

    # 5) スナップショット出力
    label = human_label(args.index_key)
    plot_snapshot(ser_pct, args.snapshot_png, label)

    # 6) テキスト出力
    updown = arrow_prefix(last_pct)
    text_lines = [
        f"{label} 日中スナップショット ({ser_pct.index[-1].tz_convert('Asia/Tokyo').strftime('%Y/%m/%d %H:%M')})",
        f"{updown} {last_pct:+.2f}%（基準: {args.basis}）",
        f"#{label} #日本株",
    ]
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))

    # 7) JSON 出力
    stats = {
        "index_key": args.index_key,
        "label": label,
        "pct_intraday": last_pct,
        "basis": args.basis,
        "session": {
            "start": args.session_start,
            "end": args.session_end,
            "anchor": args.day_anchor,
        },
        "updated_at": ser_pct.index[-1].tz_convert("Asia/Tokyo").isoformat(),
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] snapshot={args.snapshot_png}, text={args.out_text}, json={args.out_json}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
