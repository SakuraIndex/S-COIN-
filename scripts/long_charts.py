# scripts/long_charts.py
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d / 7d / 1m / 1y) for INDEX_KEY.

Inputs
- docs/outputs/<index>_intraday.csv  (time,value[,volume] 形式 もしくは 複数列の銘柄ワイド形式)
- docs/outputs/<index>_history.csv   (date,value)

Outputs
- docs/outputs/<index>_1d.png
- docs/outputs/<index>_7d.png
- docs/outputs/<index>_1m.png
- docs/outputs/<index>_1y.png
"""

from __future__ import annotations
import os
from typing import List, Optional

import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

# ====== Theme / Const ======
JP_TZ = pytz.timezone("Asia/Tokyo")
SESSION_START = "09:00"
SESSION_END   = "15:30"

BG = "#0E1117"
FG = "#E6E6E6"
ACCENT = "#3bd6c6"
TITLE  = "#f2b6c6"
GRID_A = 0.25
UP_COLOR = "#22c55e"   # 上昇（緑）
DN_COLOR = "#ef4444"   # 下落（赤）

matplotlib.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": FG,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "text.color": FG,
    "grid.color": FG,
    "savefig.facecolor": BG,
})

OUTPUTS_DIR = os.path.join("docs", "outputs")

# ====== Utilities ======
def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    # 小文字比較しつつ列名は元のまま残したい場面があるので、返すのはそのまま。
    return df

def _pick_time_col(cols: List[str]) -> Optional[str]:
    """候補の time 列名を柔軟に検出（英/日 / 大文字小文字混在を許容）"""
    lowers = {c.lower(): c for c in cols}
    for key in ("time", "timestamp", "datetime", "date", "日時"):
        if key in lowers:
            return lowers[key]
    # unnamed系（Excel出力など）
    for c in cols:
        lc = c.lower()
        if lc.startswith("unnamed") or "time" in lc or "date" in lc:
            return c
    return None

def _first_numeric_col_except(df: pd.DataFrame, except_col: str) -> Optional[str]:
    for c in df.columns:
        if c == except_col:
            continue
        try:
            series = pd.to_numeric(df[c], errors="coerce")
            if series.notna().sum() > 0:
                return c
        except Exception:
            continue
    return None

def _to_jst_series(s: pd.Series) -> pd.Series:
    """文字列/naive/UTC いずれでも JST tz-aware へ統一"""
    t = pd.to_datetime(s, errors="coerce", utc=True)
    # utc=True で tz 付きにならなかった場合（全て naive の可能性）
    if getattr(t.dt, "tz", None) is None:
        # naive を JST とみなしてローカライズ
        t = pd.to_datetime(s, errors="coerce").dt.tz_localize(JP_TZ).dt.tz_convert("UTC")
    return t.dt.tz_convert(JP_TZ)

def read_any_intraday(path: str) -> pd.DataFrame:
    """
    Return columns: time(JST tz-aware), value(float), volume(float; なければ 0)
    - long 形式: [time,value(,volume?)] あるいは [time,<index_name>]
    - wide 形式: [time,ticker1,ticker2,...] → 等加重平均
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "value", "volume"])

    raw = pd.read_csv(path, dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = _lower(raw.copy())
    # #コメント列の除去
    drop_cols = [c for c in df.columns if str(c).strip().startswith("#")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    tcol = _pick_time_col(df.columns.tolist())
    if tcol is None:
        # time列が無い→描けない
        return pd.DataFrame(columns=["time", "value", "volume"])

    # 値列の推定
    vcol = None
    volcol = None
    for c in df.columns:
        lc = c.lower()
        if c == tcol:
            continue
        if lc in ("value", "index", "score") or ("value" in lc):
            vcol = c
        if lc == "volume" or ("volume" in lc):
            volcol = c

    # time → JST tz-aware
    t = _to_jst_series(df[tcol])

    out = pd.DataFrame({"time": t})

    if vcol is None:
        # 候補の最初の数値列を値とみなす（S-COIN+ など [Datetime, S-COIN+]）
        vcol = _first_numeric_col_except(df, tcol)

    if vcol is not None:
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
        if volcol and volcol in df.columns:
            out["volume"] = pd.to_numeric(df[volcol], errors="coerce")
        else:
            out["volume"] = 0.0
    else:
        # wide → 等加重平均
        num_cols = []
        for c in df.columns:
            if c == tcol:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                num_cols.append(c)
        if not num_cols:
            return pd.DataFrame(columns=["time", "value", "volume"])
        vals = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        out["value"] = vals.mean(axis=1)
        out["volume"] = 0.0

    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def clamp_session_on_last_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    直近日のタイムスタンプ日付を基準に、取引セッション(09:00-15:30 JST)で切り出す。
    それでも 0 件なら、その日の全データ（時間帯不問）をフォールバック表示。
    """
    if df.empty or "time" not in df.columns:
        return df

    t_all = pd.to_datetime(df["time"])
    # tz-aware 確保
    if t_all.dt.tz is None:
        t_all = t_all.dt.tz_localize(JP_TZ)

    last_ts = t_all.max().astimezone(JP_TZ)
    last_day = last_ts.normalize()

    start = pd.Timestamp(f"{last_day.date()} {SESSION_START}", tz=JP_TZ)
    end   = pd.Timestamp(f"{last_day.date()} {SESSION_END}",   tz=JP_TZ)
    m = (t_all >= start) & (t_all <= end)
    cut = df.loc[m].reset_index(drop=True)
    if len(cut) > 0:
        return cut

    # セッション時間に合致しない場合 → 同一カレンダーデイだけ採用（フォールバック）
    m2 = (t_all.dt.date == last_day.date())
    cut2 = df.loc[m2].reset_index(drop=True)
    return cut2 if len(cut2) > 0 else df.tail(480).reset_index(drop=True)  # 最後の数時間を保険で

def resample_minutes(df: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.set_index("time").sort_index()
    out = tmp[["value"]].resample(rule).mean()
    out["value"] = out["value"].interpolate(limit_direction="both")
    out["volume"] = 0.0
    return out.reset_index()

def read_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "value"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    # 列名ゆるく
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date") or cols.get("日時") or list(df.columns)[0]
    vcol = cols.get("value") or _first_numeric_col_except(df, dcol) or list(df.columns)[-1]
    out = pd.DataFrame({
        "date": pd.to_datetime(df[dcol], errors="coerce"),
        "value": pd.to_numeric(df[vcol], errors="coerce")
    })
    return out.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

def _decorate(ax, title: str, xl: str, yl: str):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(True, alpha=GRID_A)
    for sp in ax.spines.values():
        sp.set_color(FG)

def _save(fig, path: str):
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

# ====== Main ======
def main():
    index_key = os.environ.get("INDEX_KEY", "").strip().lower() or \
                os.path.basename(os.getcwd()).strip().lower()
    index_name = index_key.upper().replace("_", "")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    intraday_csv = os.path.join(OUTPUTS_DIR, f"{index_key}_intraday.csv")
    history_csv  = os.path.join(OUTPUTS_DIR, f"{index_key}_history.csv")

    # ---- 1d ----
    try:
        i = read_any_intraday(intraday_csv)
        i = clamp_session_on_last_day(i)     # ← 直近日のセッションで切り出し（A の対応）
        i = resample_minutes(i, "1min")
    except Exception as e:
        print(f"[WARN] intraday load failed: {e}")
        i = pd.DataFrame(columns=["time", "value", "volume"])

    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    _decorate(ax, f"{index_name} (1d)", "Time", "Index Value")
    if not i.empty:
        first_v = float(i["value"].iloc[0])
        last_v  = float(i["value"].iloc[-1])
        color = UP_COLOR if last_v >= first_v else DN_COLOR
        ax.plot(i["time"], i["value"], linewidth=2.4, color=color)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", alpha=0.6)
    _save(fig, os.path.join(OUTPUTS_DIR, f"{index_key}_1d.png"))

    # ---- 7d / 1m / 1y ----
    h = read_history(history_csv)

    def plot_hist(tail_n: int, label: str, out: str):
        fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
        _decorate(ax, f"{index_name} ({label})", "Date", "Index Value")
        hh = h.tail(tail_n)
        if len(hh) >= 2:
            ax.plot(hh["date"], hh["value"], linewidth=2.2, color=ACCENT)
        elif len(hh) == 1:
            ax.plot(hh["date"], hh["value"], marker="o", markersize=6, linewidth=0, color=ACCENT)
            y = hh["value"].iloc[0]
            ax.set_ylim(y - 0.1, y + 0.1)
            ax.text(0.5, 0.5, "Only 1 point (need ≥ 2)", transform=ax.transAxes,
                    ha="center", va="center", alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", alpha=0.5)
        _save(fig, os.path.join(OUTPUTS_DIR, out))

    plot_hist(7,   "7d", f"{index_key}_7d.png")
    plot_hist(30,  "1m", f"{index_key}_1m.png")
    plot_hist(365, "1y", f"{index_key}_1y.png")

    # 実行記録
    with open(os.path.join(OUTPUTS_DIR, "_last_run.txt"), "w") as f:
        f.write(pd.Timestamp.now(tz=JP_TZ).isoformat())

if __name__ == "__main__":
    main()
