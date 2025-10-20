#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_intraday_post.py

CSV（Intraday）から、
- スナップショット画像（黒背景・枠なし）
- SNS 用テキスト（当日スナップショット + 基準表記）
- stats JSON（%/基準/セッション/更新時刻）

を出力するユーティリティ。

主な特徴
- 値スケール自動判定（すでに % 系列なら過剰換算を防止）
- 列名の柔軟検出（index_key が見つからない場合は 2 列目を採用）
- JST タイムゾーンでのセッション（時刻）フィルタ
- 黒背景・白縁なしのスナップショットを生成
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# ==========
# 定数
# ==========
JST = "Asia/Tokyo"
DATETIME_COL_CANDIDATES = ["Datetime", "datetime", "date", "time", "timestamp"]


# ==========
# 引数
# ==========
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate intraday snapshot & post text")
    p.add_argument("--index-key", required=True, help="index key (e.g. scoin_plus)")
    p.add_argument("--csv", required=True, help="intraday CSV path")
    p.add_argument("--out-json", required=True, help="stats JSON path")
    p.add_argument("--out-text", required=True, help="post text path")
    p.add_argument("--snapshot-png", required=True, help="snapshot PNG path")

    p.add_argument("--session-start", required=True, help="HH:MM JST (e.g. 09:00)")
    p.add_argument("--session-end", required=True, help="HH:MM JST (e.g. 15:30)")
    p.add_argument("--day-anchor", required=True, help="HH:MM JST for label (e.g. 09:00)")
    p.add_argument("--basis", required=True, help="basis label (e.g. open@09:00)")

    return p


# ==========
# モデル
# ==========
@dataclass
class SessionDef:
    start_h: int
    start_m: int
    end_h: int
    end_m: int
    anchor_h: int
    anchor_m: int

    @staticmethod
    def parse(hhmm_start: str, hhmm_end: str, hhmm_anchor: str) -> "SessionDef":
        def _split(s: str) -> Tuple[int, int]:
            h, m = s.split(":")
            return int(h), int(m)

        sh, sm = _split(hhmm_start)
        eh, em = _split(hhmm_end)
        ah, am = _split(hhmm_anchor)
        return SessionDef(sh, sm, eh, em, ah, am)


# ==========
# ユーティリティ
# ==========
def _find_datetime_column(df: pd.DataFrame) -> str:
    # 候補優先、なければ最左列名を返す
    cols = list(df.columns)
    for c in DATETIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    return cols[0]


def _normalize_index_key(key: str) -> str:
    # 記号を落として小文字化
    return "".join(ch for ch in key if ch.isalnum() or ch == "_").lower()


def _find_value_column(df: pd.DataFrame, index_key: str, dt_col: str) -> str:
    """
    1) index_key に近い列名を探す（大文字小文字無視・記号除去）
    2) 見つからなければ Datetime 以外の 2 列目を返す
    """
    norm_key = _normalize_index_key(index_key)

    candidates = []
    for c in df.columns:
        if c == dt_col:
            continue
        norm_c = _normalize_index_key(str(c))
        if norm_c == norm_key or norm_c.endswith(norm_key) or norm_key.endswith(norm_c):
            candidates.append(c)

    if candidates:
        return candidates[0]

    # 2 列目を素直に返す（Datetime 以外）
    non_dt = [c for c in df.columns if c != dt_col]
    if not non_dt:
        raise ValueError("値の列が見つかりません")
    return non_dt[0]


def _ensure_jst_index(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """
    Datetime を tz-aware にし、JST へ変換して index 化。
    - 既に tz-aware → tz_convert(JST)
    - naive        → tz_localize('UTC') とみなし tz_convert(JST)
      （多くの CSV が UTC タイムスタンプのため）
    """
    ts = pd.to_datetime(df[dt_col], utc=False, errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(JST)
    else:
        ts = ts.dt.tz_convert(JST)
    out = df.copy()
    out.index = ts
    return out.drop(columns=[dt_col])


def _filter_session(df: pd.DataFrame, sess: SessionDef) -> pd.DataFrame:
    """
    当日（JST）の session_start〜session_end のみ抽出
    """
    if df.empty:
        return df

    # 最新データの日付（JST）をベースとする
    latest_day = df.index.tz_convert(JST)[-1].date()

    start_ts = pd.Timestamp(
        year=latest_day.year, month=latest_day.month, day=latest_day.day,
        hour=sess.start_h, minute=sess.start_m, tz=JST
    )
    end_ts = pd.Timestamp(
        year=latest_day.year, month=latest_day.month, day=latest_day.day,
        hour=sess.end_h, minute=sess.end_m, tz=JST
    )
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()


def _anchor_ts_for_label(sess: SessionDef, base_ts: pd.Timestamp) -> pd.Timestamp:
    """ラベル用アンカー（同日 HH:MM JST）"""
    ts = base_ts.tz_convert(JST)
    return pd.Timestamp(
        year=ts.year, month=ts.month, day=ts.day,
        hour=sess.anchor_h, minute=sess.anchor_m, tz=JST
    )


def _auto_is_percent_series(series: pd.Series) -> bool:
    """
    値スケール自動判定：
    - 平均が 50〜200 かつ 全体の振れ幅が 20 未満 → % 基準（例：100±数％のインデックス）
    """
    try:
        mean_val = float(series.mean())
        if 50 <= mean_val <= 200:
            diff = float(series.max() - series.min())
            if diff < 20:
                return True
    except Exception:
        pass
    return False


def _to_change_percent(series: pd.Series, anchor_value: float) -> pd.Series:
    """
    通常：  (value / anchor) - 1) * 100
    既に % 系列と推定： series からアンカーとの差分 (%)
    """
    if _auto_is_percent_series(series):
        # 100 を基準とした % 変化系列と見なす → 初値との差分
        print("[INFO] Series looks already in percent scale. Using relative (%) delta from anchor.")
        return series - series.iloc[0]

    return (series / anchor_value - 1.0) * 100.0


def load_intraday(csv_path: str, index_key: str) -> Tuple[pd.DataFrame, str]:
    """
    CSV を読み、(df, value_col) を返す。
    - Datetime 列は自動検出＆JST index 化
    - 値列は index_key に近い列名 or Datetime 以外の 2 列目
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV が空です")

    dt_col = _find_datetime_column(df)
    df = _ensure_jst_index(df, dt_col)

    val_col = _find_value_column(df, index_key=index_key, dt_col=dt_col)
    # numeric 化（不要な文字が混入しても coercion）
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])

    if df.empty:
        raise ValueError("値列が全て欠損（NaN）です")

    return df[[val_col]], val_col


# ==========
# 出力
# ==========
def save_post_text(out_path: str, label: str, pct: float, basis: str):
    line1 = f"▲ {label} 日中スナップショット"
    line2 = f"▲ {pct:.2f}%（基準: {basis}）"
    line3 = f"#{label} #日本株"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{line1}\n{line2}\n{line3}\n")


def save_stats_json(out_path: str, index_key: str, label: str, pct: float,
                    session: SessionDef, updated_at_jst: pd.Timestamp, basis: str):
    obj = {
        "index_key": index_key,
        "label": label,
        "pct_intraday": float(pct),
        "basis": basis,
        "session": {
            "start": f"{session.start_h:02d}:{session.start_m:02d}",
            "end":   f"{session.end_h:02d}:{session.end_m:02d}",
            "anchor": f"{session.anchor_h:02d}:{session.anchor_m:02d}"
        },
        "updated_at": updated_at_jst.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_snapshot_png(out_path: str, series_pct: pd.Series, label: str, ts_for_title: pd.Timestamp):
    """
    黒背景・白縁無しのスナップショットを保存。
    """
    plt.close("all")
    fig = plt.figure(figsize=(11, 6), facecolor="black")
    ax = fig.add_subplot(111, facecolor="black")

    ax.plot(series_pct.index, series_pct.values, linewidth=2.0)

    # 黒背景・白縁なし
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#cccccc")
    ax.grid(True, alpha=0.15)

    title = f"{label} Intraday Snapshot ({ts_for_title.tz_convert(JST).strftime('%Y/%m/%d %H:%M')})"
    ax.set_title(title, color="#e6e6e6")
    ax.set_xlabel("Time", color="#cccccc")
    ax.set_ylabel("Change vs Anchor (%)", color="#cccccc")

    fig.tight_layout(pad=1.2)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


# ==========
# メイン
# ==========
def main():
    ap = build_parser()
    args = ap.parse_args()

    # セッション
    try:
        sess = SessionDef.parse(args.session_start, args.session_end, args.day_anchor)
    except Exception:
        raise ValueError("セッション時刻が不正です（例: 09:00 / 15:30 / 09:00）")

    # CSV 読込
    df_all, val_col = load_intraday(args.csv, index_key=args.index_key)

    # フィルタ（当日セッション）
    df = _filter_session(df_all, sess)
    if df.empty:
        raise ValueError("セッション時間内のデータがありません")

    # ラベル・アンカー
    label = args.index_key.upper()
    now_jst = pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(JST)
    anchor_ts = _anchor_ts_for_label(sess, now_jst)

    # アンカー時刻に最も近い点を基準とする（見つからない場合は最初の点）
    if (df.index <= anchor_ts).any():
        anchor_idx = df.index[df.index <= anchor_ts][-1]
    else:
        anchor_idx = df.index[0]

    series = df[val_col].copy()
    anchor_value = float(series.loc[anchor_idx])

    # 変化率（%）に変換（%系列なら相対差、価格系列なら (value/anchor-1)*100）
    series_pct = _to_change_percent(series, anchor_value=anchor_value)

    # 最新騰落率
    pct_now = float(series_pct.iloc[-1])

    # 出力
    save_post_text(args.out_text, label=label, pct=pct_now, basis=args.basis)
    save_stats_json(
        args.out_json,
        index_key=args.index_key,
        label=label,
        pct=pct_now,
        session=sess,
        updated_at_jst=now_jst,
        basis=args.basis,
    )
    save_snapshot_png(args.snapshot_png, series_pct, label=label, ts_for_title=now_jst)

    print(f"[DONE] {label} intraday: {pct_now:.2f}% (basis={args.basis})")
    print(f" - snapshot: {args.snapshot_png}")
    print(f" - post    : {args.out_text}")
    print(f" - stats   : {args.out_json}")


if __name__ == "__main__":
    main()
