# -*- coding: utf-8 -*-
# S-COIN+ Intraday: 仮想通貨関連株を等ウェイト平均で指数化（前日終値比・5分足）
# 出力:
#   docs/outputs/scoin_plus_intraday.png
#   docs/outputs/scoin_plus_intraday.csv
#   docs/outputs/scoin_plus_post.txt

import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ====== ティッカー一覧（東証: 4桁コード + ".T"）======
# 必ずご確認ください。推定が含まれます。
TICKERS_JP: Dict[str, str] = {
    "MetaPlanet(メタプラ)": "3350.T",       # 要確認
    "リミックスポイント": "3825.T",
    "DEFコンサル":       "4833.T",       # 要確認
    "ABC(旧GFA)":       "8783.T",       # 要確認
    "堀田丸正":         "8105.T",
    "エスサイエンス":     "5721.T",       # エス・サイエンス
    "イオレ":           "2334.T",       # 要確認
    "コンヴァノ":         "6574.T",
}

OUT_DIR = "docs/outputs"
IMG_PATH = os.path.join(OUT_DIR, "scoin_plus_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "scoin_plus_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "scoin_plus_post.txt")

# ====== JST ユーティリティ ======
def jst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))

os.makedirs(OUT_DIR, exist_ok=True)

# ====== データ取得 ======
def fetch_prev_close(ticker: str) -> Optional[float]:
    """
    前日終値（daily）を取得。見つからなければ None。
    """
    d = yf.download(ticker, period="10d", interval="1d", progress=False, auto_adjust=False)
    if d is None or d.empty:
        print(f"[WARN] prev close not found: {ticker}")
        return None
    # 最終日の前日（[-2]）を基本とする。市場休場等で穴がある場合は最後の確定終値を使う。
    closes = d["Close"].dropna()
    if len(closes) < 2:
        return float(closes.iloc[-1]) if len(closes) > 0 else None
    return float(closes.iloc[-2])

def fetch_intraday_series(ticker: str) -> Optional[pd.Series]:
    """
    直近1営業日の5分足（または15分足）を取得し、Close列のSeriesを返す。
    """
    # まず 1d / 5m を試す
    for iv in ["5m", "15m", "30m", "60m"]:
        d = yf.download(ticker, period="1d", interval=iv, progress=False, auto_adjust=False)
        if d is not None and not d.empty and "Close" in d.columns:
            s = pd.to_numeric(d["Close"], errors="coerce").dropna()
            if not s.empty:
                return s
    print(f"[WARN] intraday series not found: {ticker}")
    return None

# ====== 指数の等ウェイト集計 ======
def build_equal_weight_index(tickers: Dict[str, str]) -> pd.Series:
    """
    各銘柄の(価格/前日終値-1)の時系列を作り、時間でアラインし、等ウェイト平均。
    """
    series_list: List[pd.Series] = []
    used: List[str] = []

    for name, code in tickers.items():
        prev = fetch_prev_close(code)
        intraday = fetch_intraday_series(code)
        if prev is None or intraday is None:
            print(f"[WARN] skip {name} ({code})   prev={prev}  intraday={'ok' if intraday is not None else 'none'}")
            continue
        rel = intraday / prev - 1.0
        rel.name = name
        series_list.append(rel)
        used.append(name)

    if not series_list:
        raise RuntimeError("no intraday data for any ticker.")

    df = pd.concat(series_list, axis=1).sort_index()
    # 行ごとに有効値の平均（等ウェイト）
    idx = df.mean(axis=1, skipna=True)
    idx.name = "S-COIN+"
    print(f"[INFO] used tickers: {used}")
    return idx

# ====== 描画 ======
def plot_intraday_index(idx: pd.Series, color_candles: bool = True) -> None:
    plt.close("all")
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # 基準線
    ax.axhline(0, color="#666666", linewidth=1.0)

    # 線色を陽線/陰線で切替
    last = float(idx.iloc[-1])
    line_color = "#00E5D4" if last >= 0 else "#FF4D4D"

    # プロット（%へ）
    ax.plot(idx.index, idx.values * 100.0, color=line_color, linewidth=3.0, label="S-COIN+")

    # 軸など
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.tick_params(colors="white")
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    ax.legend(loc="upper right", facecolor="black", edgecolor="#444444", labelcolor="white")

    title = f"S-COIN+ Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M')})"
    ax.set_title(title, color="white", fontsize=22, pad=14)
    fig.tight_layout()

    plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

# ====== 出力 ======
def save_outputs(idx: pd.Series) -> None:
    # CSV（%）
    out = pd.DataFrame({"ChangePct": idx * 100.0})
    out.index.name = "DateTime(JST)"
    out.to_csv(CSV_PATH, encoding="utf-8")

    # 投稿用テキスト
    last_pct = float(idx.iloc[-1] * 100.0)
    sign = "🟦" if last_pct >= 0 else "🟥"
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"{sign} S-COIN+ 日中取引（{jst_now().strftime('%Y/%m/%d %H:%M')}）\n"
            f"{last_pct:+.2f}%（前日終値比）\n"
            f"構成銘柄：{ ' ／ '.join(TICKERS_JP.keys()) }\n"
            f"#仮想通貨株 #SCOINplus #日本株 #株式市場\n"
        )
    print("✅ outputs written")

def main():
    print("[INFO] Building S-COIN+ Index ...")
    idx = build_equal_weight_index(TICKERS_JP)
    plot_intraday_index(idx)
    save_outputs(idx)

if __name__ == "__main__":
    main()
