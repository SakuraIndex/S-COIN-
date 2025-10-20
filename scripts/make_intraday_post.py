#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta, timezone, date, time as dtime

import pandas as pd
import matplotlib.pyplot as plt

JST = timezone(timedelta(hours=9))


def read_intraday(csv_path: str, index_key: str) -> pd.DataFrame:
    """
    CSV を読み込み、Datetime を UTC -> JST へ変換して DatetimeIndex にする。
    index_key 列が存在するかもチェック。
    """
    df = pd.read_csv(csv_path)
    if "Datetime" not in df.columns:
        raise ValueError("CSV に 'Datetime' 列がありません。")

    if index_key not in df.columns:
        raise ValueError(f"CSVから '{index_key}' に対応する列を特定できません。候補: {set(df.columns)}")

    # 文字列→Timestamp
    ts = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
    if ts.isna().all():
        # すべてNaT→おそらく tz-naive。UTCとしてローカライズ後にJSTへ。
        ts = pd.to_datetime(df["Datetime"], errors="coerce")
        if ts.dt.tz is None:  # 全部naive
            ts = ts.dt.tz_localize("UTC")
    # UTC -> JST
    ts = ts.dt.tz_convert(JST)

    df = df.copy()
    df.index = ts
    df = df.drop(columns=["Datetime"])
    df = df.sort_index()
    return df


def filter_session(df_jst: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    """
    JSTベースの df から JSTの時刻帯でフィルタ。端点を含む。
    """
    start = datetime.combine(date=df_jst.index[-1].date(),  # 当日でOK（indexは日中の当日データ想定）
                             time=_parse_hhmm(start_hhmm),
                             tzinfo=JST).time()
    end = datetime.combine(date=df_jst.index[-1].date(),
                           time=_parse_hhmm(end_hhmm),
                           tzinfo=JST).time()

    # between_time は time only で判定（tzは無視されるが index がJST基準なのでOK）
    out = df_jst.between_time(start_time=start, end_time=end,
                              include_start=True, include_end=True)
    return out


def _parse_hhmm(hhmm: str) -> dtime:
    return datetime.strptime(hhmm, "%H:%M").time()


def pick_anchor_value(df_jst: pd.DataFrame, index_key: str, anchor_hhmm: str) -> float:
    """
    JSTの当日 {anchor_hhmm} 時刻以上の最初の値を anchor として採用。
    ぴったりの行が無ければ >= で次のサンプル、無ければ先頭をフォールバック。
    """
    anchor_t = _parse_hhmm(anchor_hhmm)
    # 当日（dfの最終日の「日付」を使う）
    day = df_jst.index[-1].date()
    anchor_dt = datetime.combine(day, anchor_t, tzinfo=JST)

    later = df_jst.loc[df_jst.index >= anchor_dt]
    if later.empty:
        # まだアンカー時刻以前しかデータがない場合は先頭を使う
        anchor_value = df_jst.iloc[0][index_key]
    else:
        anchor_value = later.iloc[0][index_key]
    return float(anchor_value)


def compute_intraday_change(anchor_value: float, last_value: float, value_type: str) -> float:
    """
    value_type:
      - "percent": 列がすでにパーセンテージ値 → 差分（last - anchor）
      - "raw":     列がレベル値 → 比率（last/anchor - 1）*100
    """
    if value_type == "percent":
        change_pct = last_value - anchor_value
    else:
        change_pct = (last_value / anchor_value - 1.0) * 100.0
    return round(float(change_pct), 4)


def plot_snapshot(df_jst: pd.DataFrame,
                  index_key: str,
                  out_png: str,
                  session_start: str,
                  session_end: str):
    """
    黒ベース・外枠無し・シアン線・薄いグリッドのスナップショット。
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_jst.index, df_jst[index_key], linewidth=2)

    ax.set_title(f"{index_key} Intraday Snapshot ({datetime.now(JST):%Y/%m/%d %H:%M})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Change vs Anchor (%)")

    ax.grid(True, linestyle="--", alpha=0.25)

    # 外枠（spines）を消す＆線色は指定しない（黒背景に馴染ませる）
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-key", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-text", required=True)
    ap.add_argument("--snapshot-png", required=True)
    ap.add_argument("--session-start", required=True)  # "09:00"
    ap.add_argument("--session-end", required=True)    # "15:30"
    ap.add_argument("--day-anchor", required=True)     # "09:00"
    ap.add_argument("--basis", required=True)          # "open@09:00" など
    ap.add_argument("--value-type", choices=["raw", "percent"], default="raw")
    args = ap.parse_args()

    # 1) 読み込み（UTC→JST変換）
    df = read_intraday(args.csv, args.index_key)

    # 2) セッション抽出（JST）
    sess = filter_session(df, args.session_start, args.session_end)
    if sess.empty:
        raise ValueError("セッション内データがありません。")

    # 3) アンカー値を取得
    anchor_val = pick_anchor_value(sess, args.index_key, args.day_anchor)
    last_val = float(sess.iloc[-1][args.index_key])

    # 4) 騰落率（％）を計算
    change_pct = compute_intraday_change(anchor_val, last_val, args.value_type)

    # 5) JSON 出力
    now_jst = datetime.now(JST)
    payload = {
        "index_key": args.index_key,
        "label": args.index_key.upper(),
        "pct_intraday": change_pct,
        "basis": args.basis,
        "session": {
            "start": args.session_start,
            "end": args.session_end,
            "anchor": args.day_anchor,
        },
        "updated_at": now_jst.isoformat(),
    }
    # pandasで JSON を出すと順序が崩れる場合があるので素直に json.dump でもOK
    import json
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 6) 投稿テキスト
    text = (
        f"▲ {args.index_key} 日中スナップショット ({now_jst:%Y/%m/%d %H:%M})\n"
        f"{change_pct:+.2f}%（基準: {args.basis}）\n"
        f"#{args.index_key} #日本株"
    )
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write(text)

    # 7) スナップショット（チャート）
    plot_snapshot(sess, args.index_key, args.snapshot_png, args.session_start, args.session_end)


if __name__ == "__main__":
    main()
