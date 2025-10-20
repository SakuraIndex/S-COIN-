#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S-COIN+（ほか汎用）用：日中スナップショット作成 & テキスト出力
- CSVの前提： "Datetime" 列 + 指数列（例: "S-COIN+"）。値は **前日終値比（%）** の時系列。
- basis:
    * "prev_close"     -> 最新の % をそのまま採用（＝前日終値比）
    * "open@HH:MM"     -> 指定JST時刻(例09:00)の値をアンカーとし、最新との差分 (%) を採用
"""

import argparse
import json
from dataclasses import dataclass
from datetime import time
from typing import Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt

JST = "Asia/Tokyo"


@dataclass
class Session:
    start_hhmm: str  # "09:00"
    end_hhmm: str    # "15:30"

    def as_times(self) -> Tuple[time, time]:
        sh, sm = map(int, self.start_hhmm.split(":"))
        eh, em = map(int, self.end_hhmm.split(":"))
        return time(sh, sm), time(eh, em)


def norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def resolve_value_column(df: pd.DataFrame, index_key: str) -> str:
    """index_key から列名を推定。"""
    candidates = [c for c in df.columns if c != "Datetime"]
    if not candidates:
        raise ValueError("CSVに値列が見つかりません。")

    mapping = {
        "scoin_plus": "S-COIN+",
        "s-coin+": "S-COIN+",
    }

    key_n = norm(index_key)
    if key_n in mapping:
        want = mapping[key_n]
        if want in df.columns:
            return want

    by_norm = {norm(c): c for c in candidates}
    if key_n in by_norm:
        return by_norm[key_n]

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(f"CSVから '{index_key}' に対応する列を特定できません。候補: {set(candidates)}")


def to_jst(df: pd.DataFrame) -> pd.DataFrame:
    """Datetime列をUTCとして解釈→JSTへ変換し、DatetimeIndexにして返す。"""
    ts = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
    # 万一すべてNaTなら（タイムゾーン情報なし）UTCとしてローカライズしてからJSTへ
    if ts.isna().all():
        ts = pd.to_datetime(df["Datetime"], errors="coerce").dt.tz_localize("UTC")
    # Series は .dt アクセサで tz_convert
    ts = ts.dt.tz_convert(JST)

    out = df.copy()
    out["Datetime"] = ts
    out = out.set_index("Datetime").sort_index()
    return out


def filter_session(df_jst: pd.DataFrame, sess: Session) -> pd.DataFrame:
    start_t, end_t = sess.as_times()
    try:
        out = df_jst.between_time(start_time=start_t, end_time=end_t)
    except Exception:
        idx = df_jst.index
        out = df_jst[(idx.time >= start_t) & (idx.time <= end_t)]
    if out.empty:
        raise ValueError("セッション内データがありません。")
    return out


def anchor_value(df_jst: pd.DataFrame, col: str, hhmm: str) -> Optional[float]:
    """JSTで hh:mm 以降最初の値を返す（なければ None）"""
    hh, mm = map(int, hhmm.split(":"))
    target = df_jst.between_time(start_time=time(hh, mm), end_time=time(23, 59))
    if target.empty:
        return None
    return float(target.iloc[0][col])


def latest_value(df_jst: pd.DataFrame, col: str) -> float:
    return float(df_jst.iloc[-1][col])


def compute_pct(df_jst: pd.DataFrame, col: str, basis: str, day_anchor: str) -> Tuple[float, str]:
    """
    basis:
      - 'prev_close'  -> 最新の % をそのまま
      - 'open@HH:MM'  -> (最新% - アンカー時刻の%) を返す
    """
    basis = basis.strip().lower()
    latest = latest_value(df_jst, col)

    if basis == "prev_close":
        return latest, "prev_close"

    if basis.startswith("open@"):
        hhmm = basis.split("@", 1)[1]
        av = anchor_value(df_jst, col, hhmm)
        if av is None:
            av = anchor_value(df_jst, col, day_anchor)
        if av is None:
            raise ValueError("アンカー時刻の値が取得できませんでした。")
        return latest - av, f"open@{hhmm}"

    # 不明な指定は prev_close にフォールバック
    return latest, "prev_close"


def make_plot(
    df_jst: pd.DataFrame,
    col: str,
    basis_label: str,
    pct_series: pd.Series,
    out_png: str,
    title_label: str,
):
    # 黒ベース・シアン線・外枠なし
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0b0b0b")
    ax.set_facecolor("#0b0b0b")

    ax.plot(pct_series.index, pct_series.values, linewidth=2.0, color="#00e5ff", label=title_label)

    # 枠線消し
    for sp in ax.spines.values():
        sp.set_visible(False)

    # 目盛とラベル色
    ax.tick_params(colors="#cfcfcf")
    ax.yaxis.label.set_color("#cfcfcf")
    ax.xaxis.label.set_color("#cfcfcf")

    # 軸ラベル
    y_label = "Change vs Anchor (%)" if basis_label.startswith("open@") else "Change vs Prev Close (%)"
    ax.set_ylabel(y_label)
    ax.set_xlabel("Time")

    # タイトル
    ts = pd.Timestamp.now(tz=JST).strftime("%Y/%m/%d %H:%M")
    ax.set_title(f"{title_label} Intraday Snapshot ({ts})", color="#ffffff", fontsize=14)

    # 凡例
    leg = ax.legend(facecolor="#1a1a1a", edgecolor="none", labelcolor="#eaeaea")
    for t in leg.get_texts():
        t.set_color("#eaeaea")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-key", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-text", required=True)
    p.add_argument("--snapshot-png", required=True)
    p.add_argument("--session-start", required=True)  # "09:00"
    p.add_argument("--session-end", required=True)    # "15:30"
    p.add_argument("--day-anchor", required=True)     # "09:00"
    p.add_argument("--basis", required=True)          # "prev_close" or "open@HH:MM"
    args = p.parse_args()

    # 読み込み
    raw = pd.read_csv(args.csv)
    if "Datetime" not in raw.columns:
        raise ValueError("CSVに 'Datetime' 列が必要です。")

    value_col = resolve_value_column(raw, args.index_key)
    df_jst = to_jst(raw[["Datetime", value_col]])

    # セッション抽出（JST）
    sess = Session(args.session_start, args.session_end)
    df_sess = filter_session(df_jst, sess)

    # 値の計算
    pct_value, basis_label = compute_pct(df_sess, value_col, args.basis, args.day_anchor)

    # プロット用系列
    if basis_label == "prev_close":
        plot_series = df_sess[value_col]
        title_label = "S-COIN+" if args.index_key.lower().startswith("scoin") else args.index_key
    else:
        hhmm = basis_label.split("@", 1)[1]
        av = anchor_value(df_sess, value_col, hhmm)
        if av is None:
            av = anchor_value(df_sess, value_col, args.day_anchor)
        if av is None:
            raise ValueError("プロット用アンカーが取得できませんでした。")
        plot_series = df_sess[value_col] - av
        title_label = "S-COIN+"

    # 図を保存
    make_plot(
        df_sess,
        value_col,
        basis_label,
        plot_series,
        args.snapshot_png,
        title_label,
    )

    # テキスト
    sign = "+" if pct_value >= 0 else ""
    now_jst = pd.Timestamp.now(tz=JST).strftime("%Y/%m/%d %H:%M")
    label_jp = ("prev_close" if basis_label == "prev_close" else basis_label)
    lines = [
        f"▲ {title_label} 日中スナップショット ({now_jst})",
        f"{sign}{pct_value:.2f}%（基準: {label_jp}）",
        "#S-COIN+ #日本株" if title_label.upper().startswith("S-COIN") else "#日本株",
    ]
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # JSON
    payload = {
        "index_key": args.index_key.upper() if args.index_key.lower().startswith("scoin") else args.index_key,
        "label": title_label.upper() if title_label else title_label,
        "pct_intraday": float(round(pct_value, 6)),
        "basis": basis_label,
        "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
        "updated_at": pd.Timestamp.now(tz=JST).isoformat(),
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
