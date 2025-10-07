# -*- coding: utf-8 -*-
"""
S-COIN+ Intraday Snapshot
仮想通貨関連株指数（東証）を日中に自動算出（等金額加重）。
"""
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ===== 銘柄一覧 =====
TICKERS = {
    "MetaPlanet": "3350.T",
    "リミックスポイント": "3825.T",
    "DEFコンサル": "4833.T",
    "ABC(旧GFA)": "8783.T",
    "堀田丸正": "8105.T",
    "エスサイエンス": "5721.T",
    "イオレ": "2334.T",
    "コンヴァノ": "6574.T",
}

OUT_DIR = "docs/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUT_DIR, "scoin_plus_intraday.csv")
IMG_PATH = os.path.join(OUT_DIR, "scoin_plus_intraday.png")
TXT_PATH = os.path.join(OUT_DIR, "scoin_plus_post.txt")


# ===== JST Utility =====
def jst_now():
    return datetime.now(timezone(timedelta(hours=9)))


# ===== データ取得補助 =====
def safe_close_series(df: pd.DataFrame) -> pd.Series:
    """DataFrameからClose列を安全にSeries化"""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    close = df.get("Close", None)
    if close is None:
        return pd.Series(dtype=float)
    if not isinstance(close, pd.Series):
        close = pd.Series(close, index=df.index)
    s = pd.to_numeric(close, errors="coerce").dropna()
    try:
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
    except Exception:
        pass
    return s


def fetch_prev_close(ticker: str):
    """前日終値を取得（安全化）"""
    df = yf.download(ticker, period="10d", interval="1d", progress=False, auto_adjust=False, prepost=False)
    s = safe_close_series(df)
    if s.empty:
        return None
    if len(s) >= 2:
        return float(s.iloc[-2])
    return float(s.iloc[-1])


def fetch_intraday_series(ticker: str):
    """1営業日分の5分足（フォールバック付き）"""
    for iv in ["5m", "15m", "30m", "60m"]:
        try:
            df = yf.download(
                ticker,
                period="1d",
                interval=iv,
                progress=False,
                auto_adjust=False,
                prepost=False,
            )
            s = safe_close_series(df)
            if not s.empty:
                return s
        except Exception as e:
            print(f"[WARN] fetch failed for {ticker} ({iv}): {e}")
            continue
    return pd.Series(dtype=float)


# ===== 指数構築 =====
def build_equal_weight_index():
    all_series = {}
    for name, code in TICKERS.items():
        prev = fetch_prev_close(code)
        intraday = fetch_intraday_series(code)
        if prev is None or intraday.empty:
            print(f"[WARN] skip {name} ({code})  prev={prev}  len={len(intraday)}")
            continue
        rel = (intraday / prev - 1.0) * 100.0  # 前日終値比（％）
        rel.name = name
        all_series[name] = rel

    if not all_series:
        raise RuntimeError("No intraday data for any ticker.")

    df = pd.concat(all_series, axis=1).ffill().dropna(how="all")
    index = df.mean(axis=1, skipna=True)
    index.name = "S-COIN+"
    return index


# ===== プロット =====
def plot_index(idx: pd.Series):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    color = "#00FFF2" if idx.iloc[-1] >= 0 else "#FF4D4D"

    ax.plot(idx.index, idx.values, color=color, linewidth=3, label="S-COIN+")
    ax.legend(facecolor="black", labelcolor="white")
    ax.axhline(0, color="#666666", linewidth=1.0)

    ax.set_title(f"S-COIN+ Intraday Snapshot ({jst_now():%Y/%m/%d %H:%M})",
                 color="white", fontsize=18)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("#444444")

    fig.tight_layout()
    plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Chart saved: {IMG_PATH}")


# ===== 出力 =====
def save_outputs(idx: pd.Series):
    idx.to_csv(CSV_PATH, encoding="utf-8-sig")
    change = float(idx.iloc[-1])
    sign = "🟩" if change >= 0 else "🟥"
    with open(TXT_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"{sign} S-COIN+ 日中取引（{jst_now():%Y/%m/%d %H:%M}）\n"
            f"{change:+.2f}%（前日終値比）\n"
            f"構成銘柄：{ ' ／ '.join(TICKERS.keys()) }\n"
            f"#仮想通貨株 #SCOINplus #日本株 #株式市場\n"
        )
    print(f"[OK] Outputs saved to {OUT_DIR}")


def main():
    print("[INFO] Building S-COIN+ intraday index ...")
    idx = build_equal_weight_index()
    plot_index(idx)
    save_outputs(idx)
    print("[DONE]")


if __name__ == "__main__":
    main()
