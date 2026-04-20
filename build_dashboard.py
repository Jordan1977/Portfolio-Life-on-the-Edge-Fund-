#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

warnings.filterwarnings("ignore")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                pass
        return super().default(obj)


def dumps_json(obj) -> str:
    return json.dumps(obj, cls=NpEncoder, ensure_ascii=False)


ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
DATA_DIR = ROOT / "data"
DOCS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

OUT_HTML = DOCS_DIR / "index.html"
OUT_JSON = DATA_DIR / "dashboard_snapshot.json"

CFG = {
    "portfolio_name": "Life on the Hedge Fund",
    "school": "Trinity College Dublin",
    "course": "Investment Analysis · Academic Portfolio",
    "school_tag": "#1 in Ireland",
    "benchmark": "QQQ",
    "secondary_benchmark": "SPY",
    "inception": "2025-03-06",
    "risk_free_rate": 0.0450,
    "annualization_factor": 252,
    "rolling_window": 30,
    "mc_seed": 42,
    "mc_paths": 600,
    "news_items": 10,
}

COLORS = {
    "bg": "#06080d",
    "panel": "#0d1420",
    "panel2": "#111b2a",
    "border": "#1d2a3f",
    "grid": "#172232",
    "text": "#dde7f3",
    "muted": "#91a4bf",
    "green": "#21d07a",
    "red": "#f45b69",
    "amber": "#ffbe55",
    "blue": "#4d8dff",
    "cyan": "#45d7ff",
    "purple": "#b085ff",
}

SCENARIOS = [
    {
        "name": "Broad market -10%",
        "type": "benchmark",
        "shock": -0.10,
        "desc": "10% QQQ drawdown mapped through portfolio beta.",
    },
    {
        "name": "Growth de-rating",
        "type": "bucket",
        "bucket": "GROWTH",
        "shock": -0.18,
        "desc": "Growth multiple compression scenario.",
    },
    {
        "name": "Speculative risk-off",
        "type": "bucket",
        "bucket": "SPECULATIVE",
        "shock": -0.25,
        "desc": "High-beta / speculative names sell off sharply.",
    },
    {
        "name": "AI multiple compression",
        "type": "theme_keyword",
        "keywords": ["AI"],
        "shock": -0.17,
        "desc": "AI narrative repricing across related positions.",
    },
    {
        "name": "Rates shock",
        "type": "custom_bucket_map",
        "mapping": {"GROWTH": -0.12, "SPECULATIVE": -0.16, "CORE": -0.06},
        "desc": "Higher real yields pressure long-duration equity exposures.",
    },
    {
        "name": "Crypto crash",
        "type": "tickers",
        "tickers": ["COIN", "MARA", "HOOD"],
        "shock": -0.28,
        "desc": "Crypto sleeve reprices sharply.",
    },
]


def esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def money(x: float, decimals: int = 0) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.{decimals}f}"


def pct(x: float, decimals: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.{decimals}f}%"


def ratio(x: float, decimals: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}x"


def tone(x: float) -> str:
    return COLORS["green"] if x >= 0 else COLORS["red"]


def to_color_list(values) -> list[str]:
    return [COLORS["green"] if float(v) >= 0 else COLORS["red"] for v in values]


def load_holdings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "ticker",
        "name",
        "quantity",
        "buy_price",
        "sector",
        "theme",
        "risk_bucket",
        "inception_date",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"holdings.csv missing columns: {sorted(missing)}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="raise")
    df["buy_price"] = pd.to_numeric(df["buy_price"], errors="raise")
    df["cost_basis"] = df["quantity"] * df["buy_price"]
    return df


def download_close_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    end_buffer = (pd.Timestamp(end) + pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end_buffer,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data")

    if isinstance(raw.columns, pd.MultiIndex):
        cols = []
        if "Close" in raw.columns.get_level_values(0):
            for t in tickers:
                if ("Close", t) in raw.columns:
                    cols.append(raw[("Close", t)].rename(t))
        elif "Close" in raw.columns.get_level_values(1):
            for t in tickers:
                if (t, "Close") in raw.columns:
                    cols.append(raw[(t, "Close")].rename(t))
        else:
            for t in tickers:
                for col in raw.columns:
                    if "close" in str(col).lower() and t.upper() in str(col).upper():
                        cols.append(raw[col].rename(t))
                        break
        if not cols:
            raise RuntimeError("Unable to extract Close prices from yfinance output")
        close = pd.concat(cols, axis=1)
    else:
        if "Close" not in raw.columns:
            raise RuntimeError("No Close column returned by yfinance")
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)

    close = close.sort_index().ffill(limit=5).dropna(how="all")

    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise RuntimeError(f"Missing downloaded series for tickers: {missing}")

    if close[CFG["benchmark"]].dropna().empty:
        raise RuntimeError(f"Benchmark {CFG['benchmark']} has no valid data")

    return close


def annualized_return(total_return: float, n_obs: int, af: int) -> float:
    if n_obs <= 0:
        return float("nan")
    return (1 + total_return) ** (af / n_obs) - 1


def downside_deviation(returns: pd.Series, minimum_acceptable_return: float, af: int) -> float:
    downside = np.minimum(returns - minimum_acceptable_return, 0)
    return float(np.sqrt(np.mean(downside**2)) * np.sqrt(af))


def omega_ratio(returns: pd.Series, minimum_acceptable_return: float) -> float:
    diff = returns - minimum_acceptable_return
    gains = diff[diff > 0].sum()
    losses = -diff[diff < 0].sum()
    return float(gains / losses) if losses > 0 else float("nan")


def capture_ratio(port_returns: pd.Series, bench_returns: pd.Series, upside: bool) -> float:
    mask = bench_returns > 0 if upside else bench_returns < 0
    if mask.sum() == 0:
        return float("nan")
    bench_mean = bench_returns[mask].mean()
    if abs(bench_mean) < 1e-12:
        return float("nan")
    return float(port_returns[mask].mean() / bench_mean)


def build_portfolio_frame(prices: pd.DataFrame, holdings: pd.DataFrame) -> dict:
    initial_aum = float(holdings["cost_basis"].sum())
    benchmark = CFG["benchmark"]
    secondary = CFG["secondary_benchmark"]

    position_mv = pd.DataFrame(index=prices.index)
    for _, row in holdings.iterrows():
        position_mv[row["ticker"]] = prices[row["ticker"]] * row["quantity"]

    nav = position_mv.sum(axis=1)

    benchmark_units = initial_aum / float(prices[benchmark].dropna().iloc[0])
    benchmark_nav = prices[benchmark] * benchmark_units

    if secondary in prices.columns and not prices[secondary].dropna().empty:
        secondary_units = initial_aum / float(prices[secondary].dropna().iloc[0])
        secondary_nav = prices[secondary] * secondary_units
        secondary_ret = secondary_nav.pct_change().fillna(0)
        secondary_b100 = (1 + secondary_ret).cumprod() * 100
    else:
        secondary_nav = pd.Series(index=prices.index, dtype=float)
        secondary_ret = pd.Series(index=prices.index, dtype=float)
        secondary_b100 = pd.Series(index=prices.index, dtype=float)

    returns = nav.pct_change().fillna(0)
    bench_returns = benchmark_nav.pct_change().fillna(0)

    return {
        "prices": prices,
        "position_mv": position_mv,
        "nav": nav,
        "benchmark_nav": benchmark_nav,
        "secondary_nav": secondary_nav,
        "returns": returns,
        "benchmark_returns": bench_returns,
        "secondary_returns": secondary_ret,
        "base100": (1 + returns).cumprod() * 100,
        "benchmark_base100": (1 + bench_returns).cumprod() * 100,
        "secondary_base100": secondary_b100,
        "drawdown": (nav / nav.cummax() - 1) * 100,
        "benchmark_drawdown": (benchmark_nav / benchmark_nav.cummax() - 1) * 100,
    }


def compute_metrics(frame: dict) -> dict:
    af = CFG["annualization_factor"]
    rf = CFG["risk_free_rate"]
    daily_rf = rf / af

    r = frame["returns"]
    b = frame["benchmark_returns"]
    n = len(r)

    total_return = float(frame["nav"].iloc[-1] / frame["nav"].iloc[0] - 1)
    bench_total_return = float(frame["benchmark_nav"].iloc[-1] / frame["benchmark_nav"].iloc[0] - 1)

    ann_return = annualized_return(total_return, n, af)
    ann_bench = annualized_return(bench_total_return, n, af)

    vol = float(r.std() * math.sqrt(af))
    bench_vol = float(b.std() * math.sqrt(af))

    ddev = downside_deviation(r, daily_rf, af)
    bench_ddev = downside_deviation(b, daily_rf, af)

    sharpe = (ann_return - rf) / vol if vol else float("nan")
    sortino = (ann_return - rf) / ddev if ddev else float("nan")

    beta = float(r.cov(b) / b.var()) if b.var() else float("nan")
    correlation = float(r.corr(b))

    alpha_vs_benchmark = total_return - bench_total_return
    jensen_alpha = (
        float(ann_return - (rf + beta * (ann_bench - rf)))
        if not math.isnan(beta)
        else float("nan")
    )

    active_returns = r - b
    tracking_error = float(active_returns.std() * math.sqrt(af))
    information_ratio = (
        float((active_returns.mean() * af) / tracking_error)
        if tracking_error
        else float("nan")
    )

    wealth = (1 + r).cumprod()
    max_drawdown = float((wealth / wealth.cummax() - 1).min())
    bench_wealth = (1 + b).cumprod()
    bench_max_drawdown = float((bench_wealth / bench_wealth.cummax() - 1).min())

    var_95 = float(np.percentile(r, 5))
    cvar_95 = float(r[r <= var_95].mean())

    treynor = (
        float((ann_return - rf) / beta)
        if not math.isnan(beta) and abs(beta) > 1e-12
        else float("nan")
    )
    omega = omega_ratio(r, daily_rf)

    upside_capture = capture_ratio(r, b, True)
    downside_capture = capture_ratio(r, b, False)

    daily_pnl = float(frame["nav"].iloc[-1] - frame["nav"].iloc[-2])
    daily_return = float(r.iloc[-1])
    total_pnl = float(frame["nav"].iloc[-1] - frame["nav"].iloc[0])

    hit_ratio = float((r > 0).mean())

    rolling_window = CFG["rolling_window"]
    rolling_vol = r.rolling(rolling_window).std() * math.sqrt(af) * 100
    rolling_beta = r.rolling(rolling_window).cov(b) / b.rolling(rolling_window).var()
    rolling_sharpe = ((r.rolling(rolling_window).mean() - daily_rf) / r.rolling(rolling_window).std()) * math.sqrt(af)

    monthly_port = (1 + r).resample("ME").prod() - 1
    monthly_bench = (1 + b).resample("ME").prod() - 1
    yearly_port = (1 + r).resample("YE").prod() - 1
    yearly_bench = (1 + b).resample("YE").prod() - 1

    return {
        "current_nav": float(frame["nav"].iloc[-1]),
        "daily_pnl": daily_pnl,
        "daily_return": daily_return,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "beta": beta,
        "correlation": correlation,
        "alpha_vs_benchmark": alpha_vs_benchmark,
        "jensen_alpha": jensen_alpha,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "hit_ratio": hit_ratio,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "downside_deviation": ddev,
        "treynor_ratio": treynor,
        "omega_ratio": omega,
        "benchmark_total_return": bench_total_return,
        "benchmark_annualized_return": ann_bench,
        "benchmark_annualized_volatility": bench_vol,
        "benchmark_max_drawdown": bench_max_drawdown,
        "upside_capture": upside_capture,
        "downside_capture": downside_capture,
        "rolling_vol": rolling_vol,
        "rolling_beta": rolling_beta,
        "rolling_sharpe": rolling_sharpe,
        "monthly_port": monthly_port,
        "monthly_bench": monthly_bench,
        "yearly_port": yearly_port,
        "yearly_bench": yearly_bench,
        "sessions": n,
        "calmar_ratio": float(ann_return / abs(max_drawdown)) if max_drawdown != 0 else float("nan"),
    }


def compute_positions(frame: dict, holdings: pd.DataFrame) -> pd.DataFrame:
    prices = frame["prices"]
    bench_returns = frame["benchmark_returns"]
    nav = float(frame["nav"].iloc[-1])
    initial_aum = float(holdings["cost_basis"].sum())

    rows = []
    for _, row in holdings.iterrows():
        ticker = row["ticker"]
        series = prices[ticker].dropna()
        last_price = float(series.iloc[-1])
        prev_price = float(series.iloc[-2]) if len(series) >= 2 else last_price
        market_value = last_price * float(row["quantity"])
        cost_basis = float(row["cost_basis"])
        pnl = market_value - cost_basis

        asset_returns = series.pct_change().dropna()
        aligned = pd.concat([asset_returns, bench_returns], axis=1, join="inner").dropna()
        aligned.columns = ["asset", "bench"]

        asset_beta = (
            float(aligned["asset"].cov(aligned["bench"]) / aligned["bench"].var())
            if len(aligned) > 10 and aligned["bench"].var()
            else float("nan")
        )

        def trailing_return(days: int) -> float:
            return float(series.iloc[-1] / series.iloc[-days - 1] - 1) if len(series) > days else float("nan")

        rows.append(
            {
                "ticker": ticker,
                "name": row["name"],
                "sector": row["sector"],
                "theme": row["theme"],
                "risk_bucket": row["risk_bucket"],
                "quantity": float(row["quantity"]),
                "buy_price": float(row["buy_price"]),
                "latest_price": last_price,
                "market_value": market_value,
                "pnl": pnl,
                "return": pnl / cost_basis,
                "weight": market_value / nav,
                "contribution": pnl / initial_aum,
                "return_1d": last_price / prev_price - 1,
                "return_5d": trailing_return(5),
                "return_1m": trailing_return(21),
                "beta": asset_beta,
            }
        )

    return pd.DataFrame(rows).sort_values("market_value", ascending=False).reset_index(drop=True)


def compute_structure(positions: pd.DataFrame) -> dict:
    sector = (
        positions.groupby("sector", as_index=False)
        .agg(weight=("weight", "sum"), market_value=("market_value", "sum"), pnl=("pnl", "sum"), count=("ticker", "count"))
        .sort_values("weight", ascending=False)
    )

    theme = (
        positions.groupby("theme", as_index=False)
        .agg(weight=("weight", "sum"), market_value=("market_value", "sum"), pnl=("pnl", "sum"), count=("ticker", "count"))
        .sort_values("weight", ascending=False)
    )

    bucket = (
        positions.groupby("risk_bucket", as_index=False)
        .agg(weight=("weight", "sum"), market_value=("market_value", "sum"), pnl=("pnl", "sum"), count=("ticker", "count"))
        .sort_values("weight", ascending=False)
    )

    hhi = float((positions["weight"] ** 2).sum())
    effective_n = float(1 / hhi) if hhi else float("nan")
    top_5_weight = float(positions["weight"].head(5).sum())

    return {
        "sector": sector,
        "theme": theme,
        "bucket": bucket,
        "hhi": hhi,
        "effective_n": effective_n,
        "top_5_weight": top_5_weight,
    }


def build_daily_ledger(frame: dict) -> pd.DataFrame:
    nav = frame["nav"]
    ledger = pd.DataFrame(
        {
            "date": nav.index,
            "nav": nav.values,
            "daily_pnl": nav.diff().fillna(0).values,
            "daily_return": frame["returns"].values,
            "benchmark_return": frame["benchmark_returns"].values,
            "active_return": (frame["returns"] - frame["benchmark_returns"]).values,
            "drawdown": (frame["drawdown"] / 100).values,
        }
    )
    return ledger.tail(40).iloc[::-1].reset_index(drop=True)


def build_monthly_heatmap(monthly_returns: pd.Series) -> pd.DataFrame:
    df = monthly_returns.to_frame("return")
    df["year"] = df.index.year
    df["month"] = df.index.strftime("%b")
    heatmap = df.pivot(index="year", columns="month", values="return")
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return heatmap.reindex(columns=[m for m in order if m in heatmap.columns])


def build_stress_tests(positions: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    nav = float(positions["market_value"].sum())
    beta = metrics["beta"]
    rows = []

    for scenario in SCENARIOS:
        shocked = positions[["ticker", "market_value", "risk_bucket", "theme"]].copy()
        shocked["shock"] = 0.0

        if scenario["type"] == "benchmark":
            shocked["shock"] = beta * scenario["shock"]
        elif scenario["type"] == "bucket":
            shocked.loc[shocked["risk_bucket"] == scenario["bucket"], "shock"] = scenario["shock"]
        elif scenario["type"] == "theme_keyword":
            mask = shocked["theme"].str.contains("|".join(scenario["keywords"]), case=False, na=False)
            shocked.loc[mask, "shock"] = scenario["shock"]
        elif scenario["type"] == "tickers":
            shocked.loc[shocked["ticker"].isin(scenario["tickers"]), "shock"] = scenario["shock"]
        elif scenario["type"] == "custom_bucket_map":
            shocked["shock"] = shocked["risk_bucket"].map(scenario["mapping"]).fillna(0.0)

        pnl_impact = float((shocked["market_value"] * shocked["shock"]).sum())
        rows.append(
            {
                "scenario": scenario["name"],
                "description": scenario["desc"],
                "pnl_impact": pnl_impact,
                "return_impact": pnl_impact / nav,
                "nav_after": nav + pnl_impact,
            }
        )

    return pd.DataFrame(rows).sort_values("pnl_impact")


def build_forecast(frame: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(CFG["mc_seed"])
    returns = frame["returns"].dropna()
    bench_returns = frame["benchmark_returns"].dropna()

    mu = float(returns.mean())
    sigma = float(returns.std())
    bench_mu = float(bench_returns.mean())
    bench_sigma = float(bench_returns.std())
    nav0 = float(frame["nav"].iloc[-1])

    horizons = {"3M": 63, "6M": 126, "12M": 252, "15Y": 252 * 15}
    summary_rows = []
    path_rows = []

    for label, horizon in horizons.items():
        shocks = np.random.normal(mu, sigma, (CFG["mc_paths"], horizon))
        wealth = nav0 * np.cumprod(1 + shocks, axis=1)
        ending = wealth[:, -1]

        summary_rows.append(
            {
                "horizon": label,
                "start_nav": nav0,
                "p05": float(np.percentile(ending, 5)),
                "p25": float(np.percentile(ending, 25)),
                "median": float(np.percentile(ending, 50)),
                "p75": float(np.percentile(ending, 75)),
                "p95": float(np.percentile(ending, 95)),
            }
        )

        steps = list(range(horizon + 1))
        nav_col = np.full((CFG["mc_paths"], 1), nav0)
        all_paths = np.hstack([nav_col, wealth])

        mc_low = np.percentile(all_paths, 5, axis=0).tolist()
        mc_high = np.percentile(all_paths, 95, axis=0).tolist()

        bull = (nav0 * np.cumprod(np.r_[1, np.repeat(bench_mu + 0.75 * bench_sigma, horizon)])).tolist()
        base = (nav0 * np.cumprod(np.r_[1, np.repeat(mu, horizon)])).tolist()
        bear = (nav0 * np.cumprod(np.r_[1, np.repeat(mu - 0.75 * sigma, horizon)])).tolist()

        for i, step in enumerate(steps):
            path_rows.append(
                {
                    "horizon": label,
                    "step": step,
                    "bull": bull[i],
                    "base": base[i],
                    "bear": bear[i],
                    "mc_low": mc_low[i],
                    "mc_high": mc_high[i],
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(path_rows)


def build_news(holdings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in holdings["ticker"].tolist():
        try:
            tk = yf.Ticker(ticker)
            news = getattr(tk, "news", None) or []
            for item in news[:5]:
                content = item.get("content", {}) if isinstance(item, dict) else {}
                title = content.get("title") or item.get("title")
                url = content.get("canonicalUrl", {}).get("url") or item.get("link") or item.get("url")
                source = content.get("provider", {}).get("displayName") or item.get("publisher") or "Yahoo Finance"
                published = content.get("pubDate") or item.get("providerPublishTime")

                if isinstance(published, str):
                    published = pd.to_datetime(published, utc=True, errors="coerce")
                elif published is not None:
                    published = pd.to_datetime(published, unit="s", utc=True, errors="coerce")
                else:
                    published = pd.NaT

                if title and url:
                    rows.append(
                        {
                            "ticker": ticker,
                            "title": title,
                            "source": source,
                            "published_at": published,
                            "url": url,
                        }
                    )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["ticker", "title", "source", "published_at", "url"])

    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.sort_values("published_at", ascending=False)
    df["_norm"] = df["title"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.drop_duplicates(subset=["_norm"]).drop(columns=["_norm"])
    return df.head(CFG["news_items"]).reset_index(drop=True)


def build_narrative(metrics: dict, positions: pd.DataFrame, structure: dict, stress: pd.DataFrame) -> list[tuple[str, str]]:
    top = positions.nlargest(3, "contribution")
    bottom = positions.nsmallest(3, "contribution")
    largest_sector = structure["sector"].iloc[0]
    worst_stress = stress.iloc[0]

    return [
        (
            "Portfolio DNA",
            f"Concentrated US equity portfolio with a deliberate high-conviction, thematic growth bias. "
            f"Top 5 positions represent {structure['top_5_weight'] * 100:.1f}% of NAV. "
            f"HHI is {structure['hhi']:.3f}, implying an effective position count of {structure['effective_n']:.1f}.",
        ),
        (
            "What worked",
            f"Performance has been led by {', '.join(top['ticker'].tolist())}. "
            f"Main detractors were {', '.join(bottom['ticker'].tolist())}.",
        ),
        (
            "Risk lens",
            f"Portfolio beta is {metrics['beta']:.2f}x versus {CFG['benchmark']}, with annualized volatility at "
            f"{metrics['annualized_volatility'] * 100:.1f}% and max drawdown at {metrics['max_drawdown'] * 100:.1f}%. "
            f"Most severe stress test is '{worst_stress['scenario']}' with an estimated NAV hit of {worst_stress['return_impact'] * 100:.1f}%.",
        ),
        (
            "Benchmark lens",
            f"Since inception, alpha versus {CFG['benchmark']} is {metrics['alpha_vs_benchmark'] * 100:+.1f}%. "
            f"Information ratio is {metrics['information_ratio']:.2f}. Upside capture is {metrics['upside_capture']:.2f}x and downside capture is {metrics['downside_capture']:.2f}x.",
        ),
        (
            "Concentration lens",
            f"The largest sector exposure is {largest_sector['sector']} at {largest_sector['weight'] * 100:.1f}% of NAV. "
            f"No rebalancing means winners are allowed to drift higher in weight over time.",
        ),
    ]


def chart_layout(title: str, height: int = 360) -> dict:
    return {
        "title": {"text": title, "x": 0.01, "font": {"size": 14, "color": COLORS["text"]}},
        "paper_bgcolor": COLORS["bg"],
        "plot_bgcolor": COLORS["panel"],
        "margin": {"l": 55, "r": 20, "t": 55, "b": 45},
        "font": {"family": "Inter, Arial, sans-serif", "size": 11, "color": COLORS["text"]},
        "xaxis": {"gridcolor": COLORS["grid"], "zeroline": False, "showline": False, "tickfont": {"color": COLORS["muted"]}},
        "yaxis": {"gridcolor": COLORS["grid"], "zeroline": False, "showline": False, "tickfont": {"color": COLORS["muted"]}},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.01},
        "hovermode": "x unified",
        "height": height,
    }


def make_charts(frame: dict, metrics: dict, positions: pd.DataFrame, structure: dict, heatmap: pd.DataFrame, stress: pd.DataFrame, forecast_paths: pd.DataFrame) -> dict:
    charts = {}

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["base100"].index, y=frame["base100"], name=CFG["portfolio_name"], line={"color": COLORS["green"], "width": 2.5}))
    fig.add_trace(go.Scatter(x=frame["benchmark_base100"].index, y=frame["benchmark_base100"], name=CFG["benchmark"], line={"color": COLORS["blue"], "width": 2}))
    if not frame["secondary_base100"].empty:
        fig.add_trace(go.Scatter(x=frame["secondary_base100"].index, y=frame["secondary_base100"], name=CFG["secondary_benchmark"], line={"color": COLORS["purple"], "width": 1.8, "dash": "dot"}))
    fig.update_layout(**chart_layout("Portfolio vs Benchmark — Base 100", 400))
    charts["performance_chart"] = fig.to_plotly_json()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["drawdown"].index, y=frame["drawdown"], name=CFG["portfolio_name"], fill="tozeroy", line={"color": COLORS["red"], "width": 2}))
    fig.add_trace(go.Scatter(x=frame["benchmark_drawdown"].index, y=frame["benchmark_drawdown"], name=CFG["benchmark"], line={"color": COLORS["blue"], "width": 1.8}))
    fig.update_layout(**chart_layout("Drawdown from Peak (%)"))
    charts["drawdown_chart"] = fig.to_plotly_json()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics["rolling_vol"].index, y=metrics["rolling_vol"], name="Rolling vol", line={"color": COLORS["amber"], "width": 2}))
    fig.update_layout(**chart_layout("Rolling 30-Day Volatility (Annualized %)"))
    charts["rolling_vol_chart"] = fig.to_plotly_json()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics["rolling_beta"].index, y=metrics["rolling_beta"], name="Rolling beta", line={"color": COLORS["cyan"], "width": 2}))
    fig.add_hline(y=1.0, line={"color": COLORS["muted"], "dash": "dot"})
    fig.update_layout(**chart_layout(f"Rolling 30-Day Beta vs {CFG['benchmark']}"))
    charts["rolling_beta_chart"] = fig.to_plotly_json()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics["rolling_sharpe"].index, y=metrics["rolling_sharpe"], name="Rolling Sharpe", line={"color": COLORS["green"], "width": 2}))
    fig.update_layout(**chart_layout("Rolling 30-Day Sharpe"))
    charts["rolling_sharpe_chart"] = fig.to_plotly_json()

    monthly_df = pd.DataFrame({"date": metrics["monthly_port"].index, "value": metrics["monthly_port"].values * 100})
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_df["date"], y=monthly_df["value"], name=CFG["portfolio_name"], marker_color=to_color_list(monthly_df["value"])))
    fig.add_trace(go.Scatter(x=metrics["monthly_bench"].index, y=metrics["monthly_bench"].values * 100, name=CFG["benchmark"], line={"color": COLORS["blue"], "width": 2}))
    fig.update_layout(**chart_layout("Monthly Returns (%)"))
    charts["monthly_returns_chart"] = fig.to_plotly_json()

    fig = go.Figure(
        data=go.Heatmap(
            z=(heatmap * 100).values.tolist(),
            x=list(heatmap.columns),
            y=[str(y) for y in heatmap.index],
            colorscale=[[0, COLORS["red"]], [0.5, COLORS["panel2"]], [1, COLORS["green"]]],
            text=[[("" if pd.isna(v) else f"{v * 100:+.1f}%") for v in row] for row in heatmap.values],
            texttemplate="%{text}",
            hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(**chart_layout("Monthly Heatmap", 320))
    charts["heatmap_chart"] = fig.to_plotly_json()

    fig = go.Figure(
        data=[
            go.Bar(
                x=positions["weight"].head(10) * 100,
                y=positions["ticker"].head(10),
                orientation="h",
                text=[f"{w * 100:.1f}%" for w in positions["weight"].head(10)],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(**chart_layout("Top Weights (%)"))
    fig.update_yaxes(autorange="reversed")
    charts["top_weights_chart"] = fig.to_plotly_json()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=structure["sector"]["sector"].tolist(),
                values=(structure["sector"]["weight"] * 100).tolist(),
                hole=0.55,
                sort=False,
            )
        ]
    )
    fig.update_layout(**chart_layout("Sector Allocation (%)", 360))
    charts["sector_chart"] = fig.to_plotly_json()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=structure["theme"]["theme"].tolist(),
                values=(structure["theme"]["weight"] * 100).tolist(),
                hole=0.55,
                sort=False,
            )
        ]
    )
    fig.update_layout(**chart_layout("Thematic Allocation (%)", 360))
    charts["theme_chart"] = fig.to_plotly_json()

    fig = go.Figure(
        data=[
            go.Bar(
                x=positions["ticker"].tolist(),
                y=positions["pnl"].tolist(),
                marker_color=to_color_list(positions["pnl"]),
                text=[money(v, 0) for v in positions["pnl"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(**chart_layout("Position P&L Attribution ($)"))
    charts["pnl_chart"] = fig.to_plotly_json()

    fig = go.Figure(
        data=[
            go.Bar(
                x=stress["pnl_impact"].tolist(),
                y=stress["scenario"].tolist(),
                orientation="h",
                marker_color=to_color_list(stress["pnl_impact"]),
                text=[money(v, 0) for v in stress["pnl_impact"]],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(**chart_layout("Stress Tests — Estimated P&L Impact", 360))
    fig.update_yaxes(autorange="reversed")
    charts["stress_chart"] = fig.to_plotly_json()

    fp12 = forecast_paths[forecast_paths["horizon"] == "12M"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fp12["step"].tolist(), y=fp12["mc_high"].tolist(), mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fp12["step"].tolist(), y=fp12["mc_low"].tolist(), mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(77,141,255,0.15)", name="Monte Carlo 5–95%"))
    fig.add_trace(go.Scatter(x=fp12["step"].tolist(), y=fp12["bull"].tolist(), name="Bull", line={"color": COLORS["green"], "width": 2}))
    fig.add_trace(go.Scatter(x=fp12["step"].tolist(), y=fp12["base"].tolist(), name="Base", line={"color": COLORS["blue"], "width": 2}))
    fig.add_trace(go.Scatter(x=fp12["step"].tolist(), y=fp12["bear"].tolist(), name="Bear", line={"color": COLORS["red"], "width": 2}))
    fig.update_layout(**chart_layout("12M Scenario Envelope — Model-Based Paths", 360))
    charts["forecast_chart"] = fig.to_plotly_json()

    return charts


def generate_html(holdings: pd.DataFrame, metrics: dict, positions: pd.DataFrame, structure: dict, ledger: pd.DataFrame, stress: pd.DataFrame, forecast_summary: pd.DataFrame, news: pd.DataFrame, narrative: list[tuple[str, str]], charts: dict) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    initial_aum = float(holdings["cost_basis"].sum())
    charts_json = dumps_json(charts)

    narrative_html = "".join(
        f"<div class='intel-block'><h3>{esc(title)}</h3><p>{esc(body)}</p></div>"
        for title, body in narrative
    )

    positions_rows = "".join(
        f"<tr>"
        f"<td>{row['ticker']}</td>"
        f"<td>{esc(row['name'])}</td>"
        f"<td>{esc(row['sector'])}</td>"
        f"<td>{esc(row['theme'])}</td>"
        f"<td>{esc(row['risk_bucket'])}</td>"
        f"<td>{row['quantity']:,.0f}</td>"
        f"<td>{money(row['buy_price'], 2)}</td>"
        f"<td>{money(row['latest_price'], 2)}</td>"
        f"<td>{money(row['market_value'], 0)}</td>"
        f"<td style='color:{tone(row['pnl'])}'>{money(row['pnl'], 0)}</td>"
        f"<td>{pct(row['return'] * 100, 1)}</td>"
        f"<td>{row['weight'] * 100:.1f}%</td>"
        f"<td>{pct(row['contribution'] * 100, 1)}</td>"
        f"</tr>"
        for _, row in positions.iterrows()
    )

    sector_rows = "".join(
        f"<tr>"
        f"<td>{esc(row['sector'])}</td>"
        f"<td>{row['weight'] * 100:.1f}%</td>"
        f"<td>{money(row['market_value'], 0)}</td>"
        f"<td style='color:{tone(row['pnl'])}'>{money(row['pnl'], 0)}</td>"
        f"<td>{int(row['count'])}</td>"
        f"</tr>"
        for _, row in structure["sector"].iterrows()
    )

    ledger_rows = "".join(
        f"<tr>"
        f"<td>{pd.to_datetime(row['date']).strftime('%Y-%m-%d')}</td>"
        f"<td>{money(row['nav'], 0)}</td>"
        f"<td style='color:{tone(row['daily_pnl'])}'>{money(row['daily_pnl'], 0)}</td>"
        f"<td>{pct(row['daily_return'] * 100, 2)}</td>"
        f"<td>{pct(row['benchmark_return'] * 100, 2)}</td>"
        f"<td>{pct(row['active_return'] * 100, 2)}</td>"
        f"<td>{pct(row['drawdown'] * 100, 2)}</td>"
        f"</tr>"
        for _, row in ledger.iterrows()
    )

    stress_rows = "".join(
        f"<tr>"
        f"<td>{esc(row['scenario'])}</td>"
        f"<td>{esc(row['description'])}</td>"
        f"<td style='color:{tone(row['pnl_impact'])}'>{money(row['pnl_impact'], 0)}</td>"
        f"<td>{pct(row['return_impact'] * 100, 1)}</td>"
        f"<td>{money(row['nav_after'], 0)}</td>"
        f"</tr>"
        for _, row in stress.iterrows()
    )

    forecast_rows = "".join(
        f"<tr>"
        f"<td>{esc(row['horizon'])}</td>"
        f"<td>{money(row['start_nav'], 0)}</td>"
        f"<td>{money(row['p05'], 0)}</td>"
        f"<td>{money(row['p25'], 0)}</td>"
        f"<td>{money(row['median'], 0)}</td>"
        f"<td>{money(row['p75'], 0)}</td>"
        f"<td>{money(row['p95'], 0)}</td>"
        f"</tr>"
        for _, row in forecast_summary.iterrows()
    )

    if news.empty:
        news_rows = "<tr><td colspan='4'>News retrieval skipped or unavailable. Core analytics unaffected.</td></tr>"
    else:
        news_rows = "".join(
            f"<tr>"
            f"<td>{esc(row['ticker'])}</td>"
            f"<td><a href='{esc(str(row['url']))}' target='_blank' rel='noopener noreferrer'>{esc(str(row['title']))}</a></td>"
            f"<td>{esc(str(row['source']))}</td>"
            f"<td>{pd.to_datetime(row['published_at'], utc=True, errors='coerce').strftime('%Y-%m-%d %H:%M UTC') if pd.notna(pd.to_datetime(row['published_at'], utc=True, errors='coerce')) else '—'}</td>"
            f"</tr>"
            for _, row in news.iterrows()
        )

    metrics_rows = "".join(
        [
            f"<tr><td>Current NAV</td><td>{money(metrics['current_nav'], 0)}</td></tr>",
            f"<tr><td>Daily P&L</td><td>{money(metrics['daily_pnl'], 0)}</td></tr>",
            f"<tr><td>Daily Return</td><td>{pct(metrics['daily_return'] * 100)}</td></tr>",
            f"<tr><td>Total P&L</td><td>{money(metrics['total_pnl'], 0)}</td></tr>",
            f"<tr><td>Total Return</td><td>{pct(metrics['total_return'] * 100)}</td></tr>",
            f"<tr><td>Annualized Return</td><td>{pct(metrics['annualized_return'] * 100)}</td></tr>",
            f"<tr><td>Annualized Volatility</td><td>{pct(metrics['annualized_volatility'] * 100)}</td></tr>",
            f"<tr><td>Sharpe Ratio</td><td>{metrics['sharpe_ratio']:.3f}</td></tr>",
            f"<tr><td>Sortino Ratio</td><td>{metrics['sortino_ratio']:.3f}</td></tr>",
            f"<tr><td>Max Drawdown</td><td>{pct(metrics['max_drawdown'] * 100)}</td></tr>",
            f"<tr><td>Beta</td><td>{metrics['beta']:.3f}</td></tr>",
            f"<tr><td>Correlation</td><td>{metrics['correlation']:.3f}</td></tr>",
            f"<tr><td>Alpha vs Benchmark</td><td>{pct(metrics['alpha_vs_benchmark'] * 100)}</td></tr>",
            f"<tr><td>Jensen Alpha</td><td>{pct(metrics['jensen_alpha'] * 100)}</td></tr>",
            f"<tr><td>Tracking Error</td><td>{pct(metrics['tracking_error'] * 100)}</td></tr>",
            f"<tr><td>Information Ratio</td><td>{metrics['information_ratio']:.3f}</td></tr>",
            f"<tr><td>Hit Ratio</td><td>{pct(metrics['hit_ratio'] * 100)}</td></tr>",
            f"<tr><td>VaR 95% (1D)</td><td>{pct(metrics['var_95'] * 100)}</td></tr>",
            f"<tr><td>CVaR 95% (1D)</td><td>{pct(metrics['cvar_95'] * 100)}</td></tr>",
            f"<tr><td>Downside Deviation</td><td>{pct(metrics['downside_deviation'] * 100)}</td></tr>",
            f"<tr><td>Treynor Ratio</td><td>{metrics['treynor_ratio']:.3f}</td></tr>",
            f"<tr><td>Omega Ratio</td><td>{metrics['omega_ratio']:.3f}</td></tr>",
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(CFG["portfolio_name"])} — Institutional Portfolio Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body {{
  margin: 0;
  background: {COLORS["bg"]};
  color: {COLORS["text"]};
  font-family: Inter, Arial, sans-serif;
}}
.wrap {{
  max-width: 1440px;
  margin: 0 auto;
  padding: 24px;
}}
header {{
  padding: 24px 0 16px;
  border-bottom: 1px solid {COLORS["border"]};
}}
h1 {{
  margin: 0 0 8px;
  font-size: 32px;
}}
.sub {{
  color: {COLORS["muted"]};
  line-height: 1.5;
}}
.small {{
  color: {COLORS["muted"]};
  font-size: 12px;
  margin-top: 8px;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
  margin: 20px 0;
}}
.kpi {{
  background: {COLORS["panel"]};
  border: 1px solid {COLORS["border"]};
  border-radius: 10px;
  padding: 14px;
}}
.kpi .label {{
  font-size: 12px;
  color: {COLORS["muted"]};
  text-transform: uppercase;
}}
.kpi .value {{
  font-size: 24px;
  font-weight: 700;
  margin-top: 6px;
}}
.panel {{
  background: {COLORS["panel"]};
  border: 1px solid {COLORS["border"]};
  border-radius: 10px;
  padding: 16px;
  margin: 24px 0;
}}
.charts-2 {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}}
.charts-3 {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 16px;
}}
.table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}}
.table th, .table td {{
  border-bottom: 1px solid {COLORS["border"]};
  padding: 8px;
  text-align: left;
}}
.table th {{
  color: {COLORS["muted"]};
  font-size: 12px;
  text-transform: uppercase;
}}
a {{
  color: {COLORS["blue"]};
}}
.intel-block {{
  margin-bottom: 14px;
}}
.intel-block h3 {{
  margin: 0 0 6px;
}}
.intel-block p {{
  margin: 0;
  color: {COLORS["muted"]};
}}
@media (max-width: 1100px) {{
  .grid, .charts-2, .charts-3 {{
    grid-template-columns: 1fr 1fr;
  }}
}}
@media (max-width: 760px) {{
  .grid, .charts-2, .charts-3 {{
    grid-template-columns: 1fr;
  }}
}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>{esc(CFG["portfolio_name"])}</h1>
    <div class="sub">
      {esc(CFG["school"])} · {esc(CFG["course"])} · Concentrated US equity · Initial AUM ~ {money(initial_aum, 0)} ·
      Benchmark {esc(CFG["benchmark"])} · Secondary benchmark {esc(CFG["secondary_benchmark"])} · No rebalancing since inception
    </div>
    <div class="small">
      Generated {generated_at} · Python is the single source of truth · Static HTML · No browser-side fetching for core analytics
    </div>
  </header>

  <div class="grid">
    <div class="kpi"><div class="label">Current NAV</div><div class="value">{money(metrics["current_nav"], 0)}</div></div>
    <div class="kpi"><div class="label">Daily P&L</div><div class="value">{money(metrics["daily_pnl"], 0)}</div></div>
    <div class="kpi"><div class="label">Total Return</div><div class="value">{pct(metrics["total_return"] * 100)}</div></div>
    <div class="kpi"><div class="label">Sharpe</div><div class="value">{metrics["sharpe_ratio"]:.3f}</div></div>
    <div class="kpi"><div class="label">Beta vs QQQ</div><div class="value">{metrics["beta"]:.3f}x</div></div>
  </div>

  <section class="panel">
    <h2>Portfolio Intelligence</h2>
    {narrative_html}
  </section>

  <section class="charts-2">
    <div class="panel"><h2>Performance</h2><div id="performance_chart"></div></div>
    <div class="panel"><h2>Drawdown</h2><div id="drawdown_chart"></div></div>
    <div class="panel"><h2>Rolling Volatility</h2><div id="rolling_vol_chart"></div></div>
    <div class="panel"><h2>Rolling Beta</h2><div id="rolling_beta_chart"></div></div>
    <div class="panel"><h2>Rolling Sharpe</h2><div id="rolling_sharpe_chart"></div></div>
    <div class="panel"><h2>Scenario Forecast</h2><div id="forecast_chart"></div></div>
  </section>

  <section class="charts-3">
    <div class="panel"><h2>Monthly Returns</h2><div id="monthly_returns_chart"></div></div>
    <div class="panel"><h2>Monthly Heatmap</h2><div id="heatmap_chart"></div></div>
    <div class="panel"><h2>Top Weights</h2><div id="top_weights_chart"></div></div>
    <div class="panel"><h2>Sector Allocation</h2><div id="sector_chart"></div></div>
    <div class="panel"><h2>Thematic Allocation</h2><div id="theme_chart"></div></div>
    <div class="panel"><h2>P&L Attribution</h2><div id="pnl_chart"></div></div>
  </section>

  <section class="panel">
    <h2>Stress Tests</h2>
    <div id="stress_chart"></div>
    <table class="table">
      <thead><tr><th>Scenario</th><th>Description</th><th>P&L impact</th><th>Return impact</th><th>NAV after</th></tr></thead>
      <tbody>{stress_rows}</tbody>
    </table>
  </section>

  <section class="panel">
    <h2>Forecast Summary</h2>
    <table class="table">
      <thead><tr><th>Horizon</th><th>Start NAV</th><th>P05</th><th>P25</th><th>Median</th><th>P75</th><th>P95</th></tr></thead>
      <tbody>{forecast_rows}</tbody>
    </table>
  </section>

  <section class="panel">
    <h2>Metrics</h2>
    <table class="table">
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{metrics_rows}</tbody>
    </table>
  </section>

  <section class="panel">
    <h2>Positions Table</h2>
    <table class="table">
      <thead>
        <tr>
          <th>Ticker</th><th>Name</th><th>Sector</th><th>Theme</th><th>Risk Bucket</th>
          <th>Qty</th><th>Buy</th><th>Last</th><th>Market Value</th><th>P&L</th><th>Return</th><th>Weight</th><th>Contribution</th>
        </tr>
      </thead>
      <tbody>{positions_rows}</tbody>
    </table>
  </section>

  <section class="panel">
    <h2>Sector Breakdown</h2>
    <table class="table">
      <thead><tr><th>Sector</th><th>Weight</th><th>Market Value</th><th>P&L</th><th>N</th></tr></thead>
      <tbody>{sector_rows}</tbody>
    </table>
  </section>

  <section class="panel">
    <h2>Daily Ledger</h2>
    <table class="table">
      <thead><tr><th>Date</th><th>NAV</th><th>Daily P&L</th><th>Daily Return</th><th>QQQ Return</th><th>Active</th><th>Drawdown</th></tr></thead>
      <tbody>{ledger_rows}</tbody>
    </table>
  </section>

  <section class="panel">
    <h2>News</h2>
    <table class="table">
      <thead><tr><th>Ticker</th><th>Headline</th><th>Source</th><th>Published</th></tr></thead>
      <tbody>{news_rows}</tbody>
    </table>
  </section>
</div>

<script>
const CHARTS = {charts_json};
const PLOTLY_CONFIG = {{
  displayModeBar: false,
  responsive: true
}};

for (const [id, fig] of Object.entries(CHARTS)) {{
  const el = document.getElementById(id);
  if (el) {{
    Plotly.newPlot(id, fig.data, fig.layout, PLOTLY_CONFIG);
  }}
}}
</script>
</body>
</html>"""


def build_snapshot(positions: pd.DataFrame, structure: dict, ledger: pd.DataFrame, stress: pd.DataFrame, forecast_summary: pd.DataFrame, narrative: list[tuple[str, str]], metrics: dict, initial_aum: float) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "portfolio_name": CFG["portfolio_name"],
            "school": CFG["school"],
            "course": CFG["course"],
            "benchmark": CFG["benchmark"],
            "secondary_benchmark": CFG["secondary_benchmark"],
            "inception": CFG["inception"],
            "risk_free_rate": CFG["risk_free_rate"],
        },
        "overview": {
            "initial_aum": initial_aum,
            "current_nav": metrics["current_nav"],
            "daily_pnl": metrics["daily_pnl"],
            "daily_return": metrics["daily_return"],
            "total_pnl": metrics["total_pnl"],
            "total_return": metrics["total_return"],
        },
        "metrics": {
            "annualized_return": metrics["annualized_return"],
            "annualized_volatility": metrics["annualized_volatility"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "beta": metrics["beta"],
            "correlation": metrics["correlation"],
            "alpha_vs_benchmark": metrics["alpha_vs_benchmark"],
            "jensen_alpha": metrics["jensen_alpha"],
            "tracking_error": metrics["tracking_error"],
            "information_ratio": metrics["information_ratio"],
            "hit_ratio": metrics["hit_ratio"],
            "var_95": metrics["var_95"],
            "cvar_95": metrics["cvar_95"],
            "downside_deviation": metrics["downside_deviation"],
            "treynor_ratio": metrics["treynor_ratio"],
            "omega_ratio": metrics["omega_ratio"],
            "upside_capture": metrics["upside_capture"],
            "downside_capture": metrics["downside_capture"],
        },
        "positions": positions.to_dict(orient="records"),
        "sector_breakdown": structure["sector"].to_dict(orient="records"),
        "daily_ledger": ledger.to_dict(orient="records"),
        "stress_tests": stress.to_dict(orient="records"),
        "forecast_summary": forecast_summary.to_dict(orient="records"),
        "portfolio_intelligence": [{"title": title, "body": body} for title, body in narrative],
    }


def main() -> None:
    holdings = load_holdings(ROOT / "holdings.csv")
    initial_aum = float(holdings["cost_basis"].sum())

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tickers = holdings["ticker"].tolist() + [CFG["benchmark"], CFG["secondary_benchmark"]]

    prices = download_close_prices(tickers, CFG["inception"], today)
    frame = build_portfolio_frame(prices, holdings)
    metrics = compute_metrics(frame)
    positions = compute_positions(frame, holdings)
    structure = compute_structure(positions)
    ledger = build_daily_ledger(frame)
    heatmap = build_monthly_heatmap(metrics["monthly_port"])
    stress = build_stress_tests(positions, metrics)
    forecast_summary, forecast_paths = build_forecast(frame)

    try:
        news = build_news(holdings)
    except Exception:
        news = pd.DataFrame(columns=["ticker", "title", "source", "published_at", "url"])

    narrative = build_narrative(metrics, positions, structure, stress)
    charts = make_charts(frame, metrics, positions, structure, heatmap, stress, forecast_paths)

    html = generate_html(
        holdings=holdings,
        metrics=metrics,
        positions=positions,
        structure=structure,
        ledger=ledger,
        stress=stress,
        forecast_summary=forecast_summary,
        news=news,
        narrative=narrative,
        charts=charts,
    )
    OUT_HTML.write_text(html, encoding="utf-8", newline="\n")

    snapshot = build_snapshot(
        positions=positions,
        structure=structure,
        ledger=ledger,
        stress=stress,
        forecast_summary=forecast_summary,
        narrative=narrative,
        metrics=metrics,
        initial_aum=initial_aum,
    )
    OUT_JSON.write_text(dumps_json(snapshot), encoding="utf-8", newline="\n")

    print(f"Generated {OUT_HTML}")
    print(f"Generated {OUT_JSON}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Build failed: {exc}")
        traceback.print_exc()
        raise
