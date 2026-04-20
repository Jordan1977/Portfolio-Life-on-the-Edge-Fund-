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


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.Index):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                pass
        return super().default(obj)


def _dumps(obj) -> str:
    return json.dumps(obj, cls=_NpEncoder, ensure_ascii=False)


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DATA = ROOT / "data"
DOCS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

OUT_HTML = DOCS / "index.html"
OUT_JSON = DATA / "dashboard_snapshot.json"

CFG = {
    "portfolio_name": "Life on the Hedge Fund",
    "school": "Trinity College Dublin",
    "school_tag": "#1 in Ireland",
    "course": "Investment Analysis · Academic Portfolio",
    "benchmark": "QQQ",
    "bench2": "SPY",
    "inception": "2025-03-06",
    "rf": 0.0450,
    "af": 252,
    "roll_w": 30,
    "mc_seed": 42,
    "mc_paths": 600,
    "news_n": 10,
}

C = {
    "bg": "#07090F",
    "panel": "#0C1018",
    "card": "#111520",
    "card2": "#161C2A",
    "border": "#1C2840",
    "border2": "#243350",
    "green": "#00D97E",
    "blue": "#2D7DD2",
    "red": "#E8304A",
    "amber": "#F0A500",
    "purple": "#7C5CBF",
    "cyan": "#00B4D8",
    "text": "#D8E2F0",
    "muted": "#7A8FAD",
    "dim": "#3A4A65",
    "grid": "#131D30",
}

SC = {
    "AI / Semiconductors": "#2D7DD2",
    "AI / Tech Platform": "#4D8FE0",
    "AI / Defence Tech": "#7C5CBF",
    "AI / AdTech": "#9169CC",
    "AI / Voice": "#A27AD8",
    "Defense / Aerospace": "#5A76C6",
    "Space Economy": "#7F5BC9",
    "Energy Transition": "#F0A500",
    "Crypto Infrastructure": "#D96B3B",
    "Bitcoin Mining": "#C24D34",
    "Fintech / Retail": "#00D97E",
    "Mobility / Platform": "#00B4D8",
    "Social / AI Data": "#4BC6B6",
}

BUCKET_C = {
    "CORE": C["blue"],
    "GROWTH": C["green"],
    "SPECULATIVE": C["amber"],
}

SCENARIOS = [
    {"name": "Broad market -10%", "type": "benchmark", "shock": -0.10, "desc": "10% QQQ drawdown mapped through beta."},
    {"name": "Growth de-rating", "type": "bucket", "bucket": "GROWTH", "shock": -0.18, "desc": "Growth multiple compression scenario."},
    {"name": "Speculative risk-off", "type": "bucket", "bucket": "SPECULATIVE", "shock": -0.25, "desc": "High-beta / speculative names sell off sharply."},
    {"name": "AI multiple compression", "type": "theme_kw", "kws": ["AI"], "shock": -0.17, "desc": "AI narrative repricing across related positions."},
    {"name": "Rates shock", "type": "custom", "mapping": {"GROWTH": -0.12, "SPECULATIVE": -0.16, "CORE": -0.06}, "desc": "Higher real yields pressure long-duration exposures."},
    {"name": "Crypto crash", "type": "tickers", "tickers": ["COIN", "MARA", "HOOD"], "shock": -0.28, "desc": "Crypto sleeve reprices sharply."},
]


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _fc(x: float, d: int = 0) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.{d}f}"


def _fp(x: float, d: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.{d}f}%"


def _fx(x: float, d: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{d}f}x"


def _col(x: float) -> str:
    return C["green"] if x >= 0 else C["red"]


def _colors(arr, pos_col: str = None, neg_col: str = None) -> list:
    pc = pos_col or C["green"]
    nc = neg_col or C["red"]
    return [pc if float(v) >= 0 else nc for v in arr]


def load_holdings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"ticker", "name", "quantity", "buy_price", "sector", "theme", "risk_bucket", "inception_date"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"holdings.csv missing columns: {sorted(missing)}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="raise")
    df["buy_price"] = pd.to_numeric(df["buy_price"], errors="raise")
    df["cost_basis"] = df["quantity"] * df["buy_price"]
    return df


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    end_buf = (pd.Timestamp(end) + pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end_buf,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data.")

    if isinstance(raw.columns, pd.MultiIndex):
        frames = []
        lv0 = [str(v) for v in raw.columns.get_level_values(0)]
        lv1 = [str(v) for v in raw.columns.get_level_values(1)]

        if "Close" in lv0:
            for t in tickers:
                col = ("Close", t)
                if col in raw.columns:
                    frames.append(raw[col].rename(t))
        elif "Close" in lv1:
            for t in tickers:
                col = (t, "Close")
                if col in raw.columns:
                    frames.append(raw[col].rename(t))

        if not frames:
            raise RuntimeError("Unable to extract Close prices from yfinance response.")

        close = pd.concat(frames, axis=1)
    else:
        if "Close" not in raw.columns:
            raise RuntimeError("No Close column returned by yfinance.")
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)

    close = close.sort_index().ffill(limit=5).dropna(how="all")

    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise RuntimeError(f"Missing tickers in downloaded data: {missing}")

    return close


def build_frame(prices: pd.DataFrame, holdings: pd.DataFrame) -> dict:
    px = prices.copy()
    bench = CFG["benchmark"]
    bench2 = CFG["bench2"]
    init = float(holdings["cost_basis"].sum())

    pos_mv = pd.DataFrame(index=px.index)
    for _, r in holdings.iterrows():
        t = r["ticker"]
        if t in px.columns:
            pos_mv[t] = px[t] * r["quantity"]

    nav = pos_mv.sum(axis=1)

    qqq_units = init / float(px[bench].dropna().iloc[0])
    qqq_nav = px[bench] * qqq_units

    if bench2 in px.columns and not px[bench2].dropna().empty:
        spy_units = init / float(px[bench2].dropna().iloc[0])
        spy_nav = px[bench2] * spy_units
        s2ret = spy_nav.pct_change().fillna(0)
        s2cum = (1 + s2ret).cumprod()
    else:
        spy_nav = pd.Series(dtype=float, index=px.index)
        s2ret = pd.Series(dtype=float, index=px.index)
        s2cum = pd.Series(dtype=float, index=px.index)

    ret = nav.pct_change().fillna(0)
    bret = qqq_nav.pct_change().fillna(0)
    cum = (1 + ret).cumprod()
    bcum = (1 + bret).cumprod()

    dd = (nav / nav.cummax() - 1) * 100
    bdd = (qqq_nav / qqq_nav.cummax() - 1) * 100

    return {
        "prices": px,
        "pos_mv": pos_mv,
        "nav": nav,
        "bench_nav": qqq_nav,
        "bench2_nav": spy_nav,
        "ret": ret,
        "bret": bret,
        "b2ret": s2ret,
        "b100": cum * 100,
        "bb100": bcum * 100,
        "b2b100": s2cum * 100,
        "dd": dd,
        "bdd": bdd,
    }


def _ann(total: float, n: int, af: int = 252) -> float:
    if n <= 0:
        return float("nan")
    return (1 + total) ** (af / n) - 1


def _downside(r: pd.Series, mar: float, af: int) -> float:
    d = np.minimum(r - mar, 0)
    return float(np.sqrt(np.mean(d**2)) * np.sqrt(af))


def _omega(r: pd.Series, mar: float) -> float:
    d = r - mar
    g = d[d > 0].sum()
    l = -d[d < 0].sum()
    return float(g / l) if l > 0 else float("nan")


def _capture(port: pd.Series, bench: pd.Series, up: bool) -> float:
    mask = bench > 0 if up else bench < 0
    if mask.sum() == 0:
        return float("nan")
    b = bench[mask].mean()
    if abs(b) < 1e-12:
        return float("nan")
    return float(port[mask].mean() / b)


def compute_metrics(frame: dict) -> dict:
    af = CFG["af"]
    rf = CFG["rf"]
    rfd = rf / af

    r = frame["ret"]
    b = frame["bret"]
    n = len(r)

    tr = float(frame["nav"].iloc[-1] / frame["nav"].iloc[0] - 1)
    trb = float(frame["bench_nav"].iloc[-1] / frame["bench_nav"].iloc[0] - 1)
    annr = _ann(tr, n, af)
    annb = _ann(trb, n, af)

    vol = float(r.std() * math.sqrt(af))
    bvol = float(b.std() * math.sqrt(af))
    ddev = _downside(r, rfd, af)
    bddev = _downside(b, rfd, af)

    sharpe = (annr - rf) / vol if vol else float("nan")
    sortino = (annr - rf) / ddev if ddev else float("nan")
    beta = float(r.cov(b) / b.var()) if b.var() else float("nan")
    corr = float(r.corr(b))
    jalpha = float(annr - (rf + beta * (annb - rf))) if not math.isnan(beta) else float("nan")

    act = r - b
    te = float(act.std() * math.sqrt(af))
    ir = float((act.mean() * af) / te) if te else float("nan")

    wealth = (1 + r).cumprod()
    dd_s = (wealth / wealth.cummax() - 1)
    mdd = float(dd_s.min())
    bwealth = (1 + b).cumprod()
    bmdd = float((bwealth / bwealth.cummax() - 1).min())
    calmar = float(annr / abs(mdd)) if mdd != 0 else float("nan")

    var95 = float(np.percentile(r, 5))
    cvar95 = float(r[r <= var95].mean())
    skew = float(r.skew())
    kurt = float(r.kurtosis())
    treynor = float((annr - rf) / beta) if (not math.isnan(beta) and abs(beta) > 1e-12) else float("nan")
    omega = _omega(r, rfd)
    upc = _capture(r, b, True)
    dnc = _capture(r, b, False)
    hit = float((r > 0).mean())

    dpnl = float(frame["nav"].iloc[-1] - frame["nav"].iloc[-2])
    dret = float(r.iloc[-1])
    tpnl = float(frame["nav"].iloc[-1] - frame["nav"].iloc[0])

    roll = CFG["roll_w"]
    rvol = r.rolling(roll).std() * math.sqrt(af) * 100
    rbeta = r.rolling(roll).cov(b) / b.rolling(roll).var()
    rsh = ((r.rolling(roll).mean() - rfd) / r.rolling(roll).std()) * math.sqrt(af)

    mp = (1 + r).resample("ME").prod() - 1
    mb = (1 + b).resample("ME").prod() - 1
    yp = (1 + r).resample("YE").prod() - 1
    yb = (1 + b).resample("YE").prod() - 1

    return dict(
        current_nav=float(frame["nav"].iloc[-1]),
        daily_pnl=dpnl,
        daily_return=dret,
        total_pnl=tpnl,
        total_return=tr,
        bench_total_return=trb,
        alpha=tr - trb,
        ann_return=annr,
        ann_bench=annb,
        vol=vol,
        bvol=bvol,
        ddev=ddev,
        bddev=bddev,
        sharpe=sharpe,
        sortino=sortino,
        beta=beta,
        corr=corr,
        jalpha=jalpha,
        te=te,
        ir=ir,
        mdd=mdd,
        bmdd=bmdd,
        calmar=calmar,
        var95=var95,
        cvar95=cvar95,
        skew=skew,
        kurt=kurt,
        treynor=treynor,
        omega=omega,
        upc=upc,
        dnc=dnc,
        hit=hit,
        sessions=n,
        rvol=rvol,
        rbeta=rbeta,
        rsh=rsh,
        mp=mp,
        mb=mb,
        yp=yp,
        yb=yb,
    )


def compute_positions(frame: dict, holdings: pd.DataFrame) -> pd.DataFrame:
    px = frame["prices"]
    bret = frame["bret"]
    nav = float(frame["nav"].iloc[-1])
    init_aum = float(holdings["cost_basis"].sum())
    rows = []

    for _, h in holdings.iterrows():
        t = h["ticker"]
        if t not in px.columns:
            continue
        s = px[t].dropna()
        r = s.pct_change().dropna()
        p = float(s.iloc[-1])
        pp = float(s.iloc[-2]) if len(s) >= 2 else p
        mv = p * float(h["quantity"])
        cb = float(h["cost_basis"])
        pnl = mv - cb

        def trail(d: int) -> float:
            return float(s.iloc[-1] / s.iloc[-d - 1] - 1) if len(s) > d else float("nan")

        aligned = pd.concat([r, bret], axis=1, join="inner").dropna()
        aligned.columns = ["r", "b"]
        beta_i = float(aligned["r"].cov(aligned["b"]) / aligned["b"].var()) if len(aligned) > 10 and aligned["b"].var() else float("nan")

        rows.append(dict(
            ticker=t,
            name=h["name"],
            sector=h["sector"],
            theme=h["theme"],
            risk_bucket=h["risk_bucket"],
            quantity=float(h["quantity"]),
            buy_price=float(h["buy_price"]),
            latest_price=p,
            market_value=mv,
            pnl=pnl,
            ret=pnl / cb,
            weight=mv / nav,
            contribution=pnl / init_aum,
            d1=p / pp - 1,
            d5=trail(5),
            d21=trail(21),
            beta=beta_i,
        ))

    pos = pd.DataFrame(rows)
    pos = pos.sort_values("market_value", ascending=False).reset_index(drop=True)
    return pos


def compute_structure(pos: pd.DataFrame) -> dict:
    sector = pos.groupby("sector", as_index=False).agg(
        weight=("weight", "sum"),
        market_value=("market_value", "sum"),
        pnl=("pnl", "sum"),
        n=("ticker", "count"),
    ).sort_values("weight", ascending=False)

    theme = pos.groupby("theme", as_index=False).agg(
        weight=("weight", "sum"),
        market_value=("market_value", "sum"),
        pnl=("pnl", "sum"),
        n=("ticker", "count"),
    ).sort_values("weight", ascending=False)

    bucket = pos.groupby("risk_bucket", as_index=False).agg(
        weight=("weight", "sum"),
        market_value=("market_value", "sum"),
        pnl=("pnl", "sum"),
        n=("ticker", "count"),
    ).sort_values("weight", ascending=False)

    hhi = float((pos["weight"] ** 2).sum())
    eff_n = float(1 / hhi) if hhi else float("nan")
    top5 = float(pos["weight"].head(5).sum())

    return dict(sector=sector, theme=theme, bucket=bucket, hhi=hhi, eff_n=eff_n, top5=top5)


def build_ledger(frame: dict) -> pd.DataFrame:
    nav = frame["nav"]
    df = pd.DataFrame({
        "date": nav.index,
        "nav": nav.values,
        "daily_pnl": nav.diff().fillna(0).values,
        "daily_ret": frame["ret"].values,
        "bench_ret": frame["bret"].values,
        "active": (frame["ret"] - frame["bret"]).values,
        "dd": frame["dd"].values / 100,
    })
    return df.tail(40).iloc[::-1].reset_index(drop=True)


def build_heatmap(mp: pd.Series) -> pd.DataFrame:
    df = mp.to_frame("r")
    df["year"] = df.index.year
    df["month"] = df.index.strftime("%b")
    pivot = df.pivot(index="year", columns="month", values="r")
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=[m for m in order if m in pivot.columns])
    return pivot


def build_stress(pos: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    nav = pos["market_value"].sum()
    beta = metrics["beta"]
    rows = []
    for s in SCENARIOS:
        shock = pos[["ticker", "market_value", "risk_bucket", "theme"]].copy()
        shock["sh"] = 0.0

        if s["type"] == "benchmark":
            shock["sh"] = beta * s["shock"]
        elif s["type"] == "bucket":
            shock.loc[shock["risk_bucket"] == s["bucket"], "sh"] = s["shock"]
        elif s["type"] == "theme_kw":
            mask = shock["theme"].str.contains("|".join(s["kws"]), case=False, na=False)
            shock.loc[mask, "sh"] = s["shock"]
        elif s["type"] == "tickers":
            shock.loc[shock["ticker"].isin(s["tickers"]), "sh"] = s["shock"]
        elif s["type"] == "custom":
            shock["sh"] = shock["risk_bucket"].map(s["mapping"]).fillna(0.0)

        total = (shock["market_value"] * shock["sh"]).sum()
        rows.append(dict(
            scenario=s["name"],
            desc=s["desc"],
            pnl_impact=total,
            ret_impact=total / nav,
            nav_after=nav + total,
        ))
    return pd.DataFrame(rows).sort_values("pnl_impact")


def build_forecast(frame: dict):
    np.random.seed(CFG["mc_seed"])
    r = frame["ret"].dropna()
    b = frame["bret"].dropna()

    mu = float(r.mean())
    sig = float(r.std())
    bmu = float(b.mean())
    bsig = float(b.std())
    nav0 = float(frame["nav"].iloc[-1])

    horizons = {"3M": 63, "6M": 126, "12M": 252, "15Y": 252 * 15}
    sum_rows, path_rows = [], []

    for lbl, h in horizons.items():
        shocks = np.random.normal(mu, sig, (CFG["mc_paths"], h))
        wealth = nav0 * np.cumprod(1 + shocks, axis=1)
        ending = wealth[:, -1]

        sum_rows.append(dict(
            horizon=lbl,
            start_nav=nav0,
            p05=float(np.percentile(ending, 5)),
            p25=float(np.percentile(ending, 25)),
            median=float(np.percentile(ending, 50)),
            p75=float(np.percentile(ending, 75)),
            p95=float(np.percentile(ending, 95)),
        ))

        steps = list(range(h + 1))
        nav_col = np.full((CFG["mc_paths"], 1), nav0)
        all_w = np.hstack([nav_col, wealth])

        p05p = np.percentile(all_w, 5, axis=0).tolist()
        p95p = np.percentile(all_w, 95, axis=0).tolist()
        bull = (nav0 * np.cumprod(np.r_[1, np.repeat(bmu + 0.75 * bsig, h)])).tolist()
        base = (nav0 * np.cumprod(np.r_[1, np.repeat(mu, h)])).tolist()
        bear = (nav0 * np.cumprod(np.r_[1, np.repeat(mu - 0.75 * sig, h)])).tolist()

        for i, step in enumerate(steps):
            path_rows.append(dict(
                horizon=lbl,
                step=step,
                bull=bull[i],
                base=base[i],
                bear=bear[i],
                mc_low=p05p[i],
                mc_high=p95p[i],
            ))

    return pd.DataFrame(sum_rows), pd.DataFrame(path_rows)


def build_news(holdings: pd.DataFrame) -> pd.DataFrame:
    items = []
    for ticker in holdings["ticker"].tolist():
        try:
            tk = yf.Ticker(ticker)
            news = getattr(tk, "news", None) or []
            for a in news[:5]:
                ct = a.get("content", {}) if isinstance(a, dict) else {}
                ttl = ct.get("title") or a.get("title")
                url = ct.get("canonicalUrl", {}).get("url") or a.get("link") or a.get("url")
                src = ct.get("provider", {}).get("displayName") or a.get("publisher") or "Yahoo Finance"
                pub = ct.get("pubDate") or a.get("providerPublishTime")

                if isinstance(pub, str):
                    pub = pd.to_datetime(pub, utc=True, errors="coerce")
                elif pub is not None:
                    pub = pd.to_datetime(pub, unit="s", utc=True, errors="coerce")
                else:
                    pub = pd.NaT

                if ttl and url:
                    items.append(dict(
                        ticker=ticker,
                        title=ttl,
                        source=src,
                        published_at=pub,
                        url=url,
                    ))
        except Exception:
            continue

    if not items:
        return pd.DataFrame(columns=["ticker", "title", "source", "published_at", "url"])

    df = pd.DataFrame(items)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.sort_values("published_at", ascending=False)
    df["_norm"] = df["title"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.drop_duplicates(subset=["_norm"]).drop(columns=["_norm"])
    return df.head(CFG["news_n"]).reset_index(drop=True)


def build_intelligence(metrics: dict, pos: pd.DataFrame, structure: dict, stress: pd.DataFrame) -> list:
    top3 = pos.nlargest(3, "contribution")
    bot3 = pos.nsmallest(3, "contribution")
    dsec = structure["sector"].iloc[0]
    wstress = stress.iloc[0]

    return [
        ("Portfolio DNA",
         f"Concentrated US equity book with high-conviction thematic growth bias. Top 5 positions represent {structure['top5']*100:.1f}% of NAV. HHI {structure['hhi']:.3f} → effective position count {structure['eff_n']:.1f}."),
        ("What worked",
         f"Top contribution came from {', '.join(top3['ticker'].tolist())}. Main detractors were {', '.join(bot3['ticker'].tolist())}."),
        ("Risk lens",
         f"Beta {metrics['beta']:.2f}x vs {CFG['benchmark']}, annualized vol {metrics['vol']*100:.1f}%, max drawdown {metrics['mdd']*100:.1f}%. Most severe modelled shock: '{wstress['scenario']}' ({wstress['ret_impact']*100:.1f}% NAV hit)."),
        ("Benchmark lens",
         f"Alpha vs {CFG['benchmark']}: {metrics['alpha']*100:+.1f}% since inception. Information ratio {metrics['ir']:.2f}. Upside capture {metrics['upc']:.2f}x, downside capture {metrics['dnc']:.2f}x."),
        ("Concentration lens",
         f"Dominant sector: {dsec['sector']} at {dsec['weight']*100:.1f}% NAV. No rebalancing means winners are allowed to drift higher in weight over time."),
    ]


def _layout(title: str, h: int = 360) -> dict:
    return dict(
        title=dict(
            text=title,
            x=0.0,
            xanchor="left",
            font=dict(size=14, color=C["text"], family="DM Sans, Arial, sans-serif"),
        ),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["card"],
        margin=dict(l=56, r=18, t=52, b=42),
        font=dict(
            family="DM Sans, Arial, sans-serif",
            size=11,
            color=C["text"],
        ),
        xaxis=dict(
            type="date",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.045)",
            gridwidth=1,
            zeroline=False,
            showline=False,
            ticks="outside",
            ticklen=4,
            tickcolor=C["border"],
            tickfont=dict(size=10, color=C["muted"], family="JetBrains Mono, monospace"),
            tickformat="%b %Y",
            hoverformat="%d %b %Y",
            automargin=True,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.045)",
            gridwidth=1,
            zeroline=False,
            showline=False,
            ticks="outside",
            ticklen=4,
            tickcolor=C["border"],
            tickfont=dict(size=10, color=C["muted"], family="JetBrains Mono, monospace"),
            automargin=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=C["muted"], family="JetBrains Mono, monospace"),
        ),
        hoverlabel=dict(
            bgcolor=C["card2"],
            bordercolor=C["border2"],
            font=dict(color=C["text"], size=11, family="DM Sans, Arial, sans-serif"),
        ),
        hovermode="x unified",
        height=h,
    )


def make_charts(frame: dict, metrics: dict, pos: pd.DataFrame,
                structure: dict, heatmap: pd.DataFrame,
                stress: pd.DataFrame, fp: pd.DataFrame) -> dict:
    charts = {}

    portfolio_color = "#00D97E"
    benchmark_color = "#2D7DD2"
    benchmark2_color = "#7C5CBF"
    down_fill = "rgba(232,48,74,0.14)"
    green_soft = "rgba(0,217,126,0.14)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame["b100"].index,
        y=frame["b100"],
        name=CFG["portfolio_name"],
        line=dict(color=portfolio_color, width=3.4),
        hovertemplate="%{x|%d %b %Y}<br><b>Portfolio</b>: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=frame["bb100"].index,
        y=frame["bb100"],
        name=CFG["benchmark"],
        line=dict(color=benchmark_color, width=2.1),
        hovertemplate="%{x|%d %b %Y}<br><b>QQQ</b>: %{y:.2f}<extra></extra>",
    ))
    if not frame["b2b100"].empty:
        fig.add_trace(go.Scatter(
            x=frame["b2b100"].index,
            y=frame["b2b100"],
            name=CFG["bench2"],
            line=dict(color=benchmark2_color, width=1.5, dash="dot"),
            hovertemplate="%{x|%d %b %Y}<br><b>SPY</b>: %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(**_layout("Portfolio vs Benchmarks — Base 100", 420))
    fig.update_yaxes(title_text="Base 100")
    charts["perf"] = fig.to_plotly_json()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame["dd"].index,
        y=frame["dd"],
        name=CFG["portfolio_name"],
        fill="tozeroy",
        fillcolor=down_fill,
        line=dict(color=C["red"], width=2.2),
        hovertemplate="%{x|%d %b %Y}<br><b>Portfolio DD</b>: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=frame["bdd"].index,
        y=frame["bdd"],
        name=CFG["benchmark"],
        line=dict(color=benchmark_color, width=1.7),
        hovertemplate="%{x|%d %b %Y}<br><b>QQQ DD</b>: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(**_layout("Drawdown from Peak", 330))
    fig.update_yaxes(title_text="%")
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.10)", width=1))
    charts["drawdown"] = fig.to_plotly_json()

    mp = metrics["mp"]
    mdf = pd.DataFrame({"date": mp.index, "val": mp.values * 100})
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mdf["date"],
        y=mdf["val"],
        name=CFG["portfolio_name"],
        marker_color=_colors(mdf["val"], pos_col=portfolio_color, neg_col=C["red"]),
        marker_line_width=0,
        opacity=0.90,
        hovertemplate="%{x|%b %Y}<br><b>Portfolio</b>: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=metrics["mb"].index,
        y=metrics["mb"].values * 100,
        name=CFG["benchmark"],
        line=dict(color=benchmark_color, width=2.0),
        hovertemplate="%{x|%b %Y}<br><b>QQQ</b>: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(**_layout("Monthly Returns vs QQQ", 330))
    fig.update_yaxes(title_text="%")
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.10)", width=1))
    charts["monthly_returns"] = fig.to_plotly_json()

    rv = metrics["rvol"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rv.index,
        y=rv,
        name="Portfolio vol",
        line=dict(color=C["amber"], width=2.1),
        hovertemplate="%{x|%d %b %Y}<br>Vol: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(**_layout(f"Rolling {CFG['roll_w']}-Day Volatility (Ann.)", 320))
    fig.update_yaxes(title_text="%")
    charts["rolling_vol"] = fig.to_plotly_json()

    rb = metrics["rbeta"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rb.index,
        y=rb,
        name="Rolling beta",
        line=dict(color=C["cyan"], width=2.1),
        hovertemplate="%{x|%d %b %Y}<br>Beta: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line=dict(color="rgba(255,255,255,0.16)", dash="dot", width=1))
    fig.update_layout(**_layout(f"Rolling {CFG['roll_w']}-Day Beta vs {CFG['benchmark']}", 320))
    charts["rolling_beta"] = fig.to_plotly_json()

    rsh = metrics["rsh"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rsh.index,
        y=rsh,
        name="Rolling Sharpe",
        line=dict(color=portfolio_color, width=2.1),
        hovertemplate="%{x|%d %b %Y}<br>Sharpe: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.16)", dash="dot", width=1))
    fig.update_layout(**_layout(f"Rolling {CFG['roll_w']}-Day Sharpe", 320))
    charts["rolling_sharpe"] = fig.to_plotly_json()

    fig = go.Figure(data=go.Heatmap(
        z=(heatmap * 100).values.tolist(),
        x=list(heatmap.columns),
        y=[str(y) for y in heatmap.index],
        colorscale=[
            [0.0, "#6E2030"],
            [0.48, "#121926"],
            [0.50, "#161C2A"],
            [0.52, "#143124"],
            [1.0, "#00D97E"],
        ],
        zmid=0,
        text=[[("" if pd.isna(v) else f"{v*100:+.1f}%") for v in row] for row in heatmap.values],
        texttemplate="%{text}",
        textfont={"color": "#D8E2F0", "size": 11},
        hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
        xgap=1,
        ygap=1,
        colorbar=dict(
            thickness=10,
            tickfont=dict(size=10, color=C["muted"], family="JetBrains Mono, monospace"),
            outlinewidth=0,
        ),
    ))
    fig.update_layout(**_layout("Monthly Return Heatmap", 310))
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    charts["heatmap"] = fig.to_plotly_json()

    top_pos = pos.head(10).copy().sort_values("weight", ascending=True)
    fig = go.Figure(data=[go.Bar(
        x=(top_pos["weight"] * 100).tolist(),
        y=top_pos["ticker"].tolist(),
        orientation="h",
        marker_color=[BUCKET_C.get(x, C["blue"]) for x in top_pos["risk_bucket"]],
        marker_line_width=0,
        text=[f"{w*100:.1f}%" for w in top_pos["weight"]],
        textposition="outside",
        hovertemplate="%{y}<br>Weight: %{x:.2f}%<extra></extra>",
    )])
    fig.update_layout(**_layout("Top Weights", 340))
    fig.update_xaxes(title_text="% NAV")
    charts["top_weights"] = fig.to_plotly_json()

    fig = go.Figure(data=[go.Pie(
        labels=structure["sector"]["sector"].tolist(),
        values=(structure["sector"]["weight"] * 100).tolist(),
        hole=0.60,
        marker=dict(
            colors=[SC.get(s, C["blue"]) for s in structure["sector"]["sector"]],
            line=dict(width=1, color=C["bg"]),
        ),
        sort=False,
        textinfo="label+percent",
        textfont=dict(size=11),
        hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
    )])
    fig.update_layout(**_layout("Sector Allocation", 350))
    charts["sector_alloc"] = fig.to_plotly_json()

    theme_colors = [
        "#2D7DD2", "#7C5CBF", "#00D97E", "#F0A500", "#00B4D8",
        "#7A8FAD", "#95A7C2", "#4E79B7", "#6C89B9", "#8DA0C1",
        "#4DBF8B", "#D89A3C", "#8BD8F8"
    ]
    fig = go.Figure(data=[go.Pie(
        labels=structure["theme"]["theme"].tolist(),
        values=(structure["theme"]["weight"] * 100).tolist(),
        hole=0.60,
        marker=dict(
            colors=theme_colors[:len(structure["theme"])],
            line=dict(width=1, color=C["bg"]),
        ),
        sort=False,
        textinfo="label+percent",
        textfont=dict(size=11),
        hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
    )])
    fig.update_layout(**_layout("Thematic Allocation", 350))
    charts["theme_alloc"] = fig.to_plotly_json()

    pnl_pos = pos.copy().sort_values("pnl", ascending=True)
    fig = go.Figure(data=[go.Bar(
        x=pnl_pos["pnl"].tolist(),
        y=pnl_pos["ticker"].tolist(),
        orientation="h",
        marker_color=_colors(pnl_pos["pnl"], pos_col=portfolio_color, neg_col=C["red"]),
        marker_line_width=0,
        hovertemplate="%{y}<br>P&L: %{x:$,.0f}<extra></extra>",
    )])
    fig.update_layout(**_layout("Position P&L Attribution", 360))
    fig.update_xaxes(title_text="$")
    charts["pnl_attr"] = fig.to_plotly_json()

    stress_plot = stress.copy().sort_values("pnl_impact", ascending=True)
    fig = go.Figure(data=[go.Bar(
        x=stress_plot["pnl_impact"].tolist(),
        y=stress_plot["scenario"].tolist(),
        orientation="h",
        marker_color=_colors(stress_plot["pnl_impact"], pos_col=portfolio_color, neg_col=C["red"]),
        marker_line_width=0,
        hovertemplate="%{y}<br>P&L impact: %{x:$,.0f}<extra></extra>",
    )])
    fig.update_layout(**_layout("Stress Test P&L Impact", 350))
    fig.update_xaxes(title_text="$")
    charts["stress"] = fig.to_plotly_json()

    fp12 = fp[fp["horizon"] == "12M"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fp12["step"].tolist(),
        y=fp12["mc_high"].tolist(),
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fp12["step"].tolist(),
        y=fp12["mc_low"].tolist(),
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor=green_soft,
        name="MC 5–95%",
    ))
    fig.add_trace(go.Scatter(
        x=fp12["step"].tolist(),
        y=fp12["bull"].tolist(),
        name="Bull",
        line=dict(color=portfolio_color, width=2.0),
    ))
    fig.add_trace(go.Scatter(
        x=fp12["step"].tolist(),
        y=fp12["base"].tolist(),
        name="Base",
        line=dict(color=benchmark_color, width=2.0),
    ))
    fig.add_trace(go.Scatter(
        x=fp12["step"].tolist(),
        y=fp12["bear"].tolist(),
        name="Bear",
        line=dict(color=C["red"], width=2.0),
    ))
    fig.update_layout(**_layout("12M Scenario Envelope", 340))
    fig.update_xaxes(type="linear", title_text="Trading days")
    fig.update_yaxes(title_text="$")
    charts["forecast"] = fig.to_plotly_json()

    return charts


def _kpi(label: str, value: str, sub: str = "", tone: str = "b") -> str:
    return (
        f"<div class='kpi {tone}'>"
        f"<div class='kpi-label'>{_esc(label)}</div>"
        f"<div class='kpi-val'>{value}</div>"
        f"<div class='kpi-sub'>{_esc(sub)}</div>"
        f"</div>"
    )


def _risk_class(v: str) -> str:
    v = str(v).upper()
    if v == "CORE":
        return "core"
    if v == "GROWTH":
        return "growth"
    return "speculative"


def _positions_table(pos: pd.DataFrame) -> str:
    rows = []
    for _, r in pos.iterrows():
        rows.append(
            f"<tr>"
            f"<td class='td-ticker'>{r['ticker']}</td>"
            f"<td class='td-name'>{_esc(r['name'])}</td>"
            f"<td>{_esc(r['sector'])}</td>"
            f"<td>{_esc(r['theme'])}</td>"
            f"<td><span class='bucket {_risk_class(r['risk_bucket'])}'>{_esc(r['risk_bucket'])}</span></td>"
            f"<td class='num'>{r['quantity']:,.0f}</td>"
            f"<td class='num'>{_fc(r['buy_price'],2)}</td>"
            f"<td class='num'>{_fc(r['latest_price'],2)}</td>"
            f"<td class='num'>{_fc(r['market_value'],0)}</td>"
            f"<td class='num' style='color:{_col(r['pnl'])}'>{_fc(r['pnl'],0)}</td>"
            f"<td class='num' style='color:{_col(r['ret'])}'>{_fp(r['ret']*100,1)}</td>"
            f"<td class='num'>{r['weight']*100:.1f}%</td>"
            f"<td class='num' style='color:{_col(r['d1'])}'>{_fp(r['d1']*100,1)}</td>"
            f"<td class='num' style='color:{_col(r['d5'])}'>{_fp(r['d5']*100,1)}</td>"
            f"<td class='num' style='color:{_col(r['d21'])}'>{_fp(r['d21']*100,1)}</td>"
            f"<td class='num'>{r['beta']:.2f}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _metrics_table(m: dict, pos: pd.DataFrame) -> str:
    rows = [
        ("Current NAV", _fc(m["current_nav"],0), "Marked-to-market NAV."),
        ("Total P&L", _fc(m["total_pnl"],0), "Absolute P&L since inception."),
        ("Total Return", _fp(m["total_return"]*100), "Portfolio return since inception."),
        ("Benchmark Return", _fp(m["bench_total_return"]*100), f"{CFG['benchmark']} since inception."),
        ("Alpha vs Benchmark", _fp(m["alpha"]*100), "Simple excess return vs QQQ."),
        ("Annualized Return", _fp(m["ann_return"]*100), "Compounded annualized return."),
        ("Annualized Volatility", _fp(m["vol"]*100), "Realized annualized vol."),
        ("Sharpe Ratio", f"{m['sharpe']:.3f}", f"rf={CFG['rf']*100:.2f}%"),
        ("Sortino Ratio", f"{m['sortino']:.3f}", "Downside-adjusted."),
        ("Calmar Ratio", f"{m['calmar']:.3f}", "Ann. return / |max drawdown|."),
        ("Max Drawdown", _fp(m["mdd"]*100), "Peak-to-trough drawdown."),
        ("Beta", f"{m['beta']:.3f}", f"Vs {CFG['benchmark']}."),
        ("Correlation", f"{m['corr']:.3f}", f"To {CFG['benchmark']}."),
        ("Jensen Alpha", _fp(m["jalpha"]*100), "CAPM alpha."),
        ("Tracking Error", _fp(m["te"]*100), "Annualized active risk."),
        ("Information Ratio", f"{m['ir']:.3f}", "Active return / active risk."),
        ("VaR 95% (1d)", _fp(m["var95"]*100), "Historical 5th percentile."),
        ("CVaR 95% (1d)", _fp(m["cvar95"]*100), "Expected shortfall."),
        ("Downside Deviation", _fp(m["ddev"]*100), "Annualized downside vol."),
        ("Treynor Ratio", f"{m['treynor']:.3f}", "Excess return / beta."),
        ("Omega Ratio", f"{m['omega']:.3f}", "Gain/loss above rf threshold."),
        ("Upside Capture", _fx(m["upc"]), "On benchmark up days."),
        ("Downside Capture", _fx(m["dnc"]), "On benchmark down days."),
        ("Hit Ratio", _fp(m["hit"]*100), "% positive days."),
        ("Skewness", f"{m['skew']:.3f}", "Return distribution skew."),
        ("Kurtosis", f"{m['kurt']:.3f}", "Tail fatness."),
        ("Sessions", str(m["sessions"]), "Trading days since inception."),
    ]
    return "\n".join(
        f"<tr><td>{_esc(l)}</td><td class='num'>{v}</td><td>{_esc(n)}</td></tr>"
        for l, v, n in rows
    )


def _sector_table(structure: dict) -> str:
    rows = []
    for _, r in structure["sector"].iterrows():
        rows.append(
            f"<tr>"
            f"<td>{_esc(r['sector'])}</td>"
            f"<td class='num'>{r['weight']*100:.1f}%</td>"
            f"<td class='num'>{_fc(r['market_value'],0)}</td>"
            f"<td class='num' style='color:{_col(r['pnl'])}'>{_fc(r['pnl'],0)}</td>"
            f"<td class='num'>{int(r['n'])}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _ledger_table(ledger: pd.DataFrame) -> str:
    rows = []
    for _, r in ledger.iterrows():
        rows.append(
            f"<tr>"
            f"<td>{pd.to_datetime(r['date']).strftime('%Y-%m-%d')}</td>"
            f"<td class='num'>{_fc(r['nav'],0)}</td>"
            f"<td class='num' style='color:{_col(r['daily_pnl'])}'>{_fc(r['daily_pnl'],0)}</td>"
            f"<td class='num' style='color:{_col(r['daily_ret'])}'>{_fp(r['daily_ret']*100,2)}</td>"
            f"<td class='num'>{_fp(r['bench_ret']*100,2)}</td>"
            f"<td class='num' style='color:{_col(r['active'])}'>{_fp(r['active']*100,2)}</td>"
            f"<td class='num' style='color:{_col(r['dd'])}'>{_fp(r['dd']*100,2)}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _stress_table(stress: pd.DataFrame) -> str:
    rows = []
    for _, r in stress.iterrows():
        rows.append(
            f"<tr>"
            f"<td>{_esc(r['scenario'])}</td>"
            f"<td>{_esc(r['desc'])}</td>"
            f"<td class='num' style='color:{_col(r['pnl_impact'])}'>{_fc(r['pnl_impact'],0)}</td>"
            f"<td class='num' style='color:{_col(r['ret_impact'])}'>{_fp(r['ret_impact']*100,1)}</td>"
            f"<td class='num'>{_fc(r['nav_after'],0)}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _forecast_table(fs: pd.DataFrame) -> str:
    rows = []
    for _, r in fs.iterrows():
        rows.append(
            f"<tr>"
            f"<td>{_esc(r['horizon'])}</td>"
            f"<td class='num'>{_fc(r['start_nav'],0)}</td>"
            f"<td class='num'>{_fc(r['p05'],0)}</td>"
            f"<td class='num'>{_fc(r['p25'],0)}</td>"
            f"<td class='num'>{_fc(r['median'],0)}</td>"
            f"<td class='num'>{_fc(r['p75'],0)}</td>"
            f"<td class='num'>{_fc(r['p95'],0)}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _news_table(news: pd.DataFrame) -> str:
    if news.empty:
        return "<tr><td colspan='4'>News retrieval unavailable. Core analytics unaffected.</td></tr>"

    rows = []
    for _, r in news.iterrows():
        pub = pd.to_datetime(r["published_at"], utc=True, errors="coerce")
        pt = pub.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(pub) else "—"
        rows.append(
            f"<tr>"
            f"<td>{_esc(r['ticker'])}</td>"
            f"<td><a href='{_esc(str(r['url']))}' target='_blank' rel='noopener noreferrer'>{_esc(str(r['title']))}</a></td>"
            f"<td>{_esc(str(r['source']))}</td>"
            f"<td>{pt}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def generate_html(
    holdings, frame, metrics, pos, structure, ledger,
    heatmap, stress, fcast_sum, fcast_paths, news, intel, charts
) -> str:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    init_aum = float(holdings["cost_basis"].sum())
    chart_json = _dumps({k: v for k, v in charts.items()})

    top_return = metrics["total_return"] * 100
    top_alpha = metrics["alpha"] * 100

    hero = f"""
    <div id="overview" class="hero">
      <div class="hero-block">
        <div class="hero-label">Portfolio NAV</div>
        <div class="hero-val green">{_fc(metrics["current_nav"],0)}</div>
        <div class="hero-sub">from {_fc(init_aum,0)} · {metrics["sessions"]} sessions</div>
      </div>
      <div class="hero-block">
        <div class="hero-label">Total P&amp;L</div>
        <div class="hero-val {'green' if metrics['total_pnl'] >= 0 else 'red'}">{_fc(metrics["total_pnl"],0)}</div>
        <div class="hero-sub">since {CFG["inception"]}</div>
      </div>
      <div class="hero-block">
        <div class="hero-label">Total Return</div>
        <div class="hero-val {'green' if metrics['total_return'] >= 0 else 'red'}">{_fp(top_return)}</div>
        <div class="hero-sub">QQQ: {_fp(metrics["bench_total_return"]*100)} · α: {_fp(top_alpha)}</div>
      </div>
      <div class="hero-block">
        <div class="hero-label">Sharpe / Sortino</div>
        <div class="hero-val blue">{metrics["sharpe"]:.3f}</div>
        <div class="hero-sub">Sortino: {metrics["sortino"]:.3f} | Calmar: {metrics["calmar"]:.3f}</div>
      </div>
      <div class="hero-right">
        <div class="hero-pill">
          <span class="pill-label">Beta</span>
          <span class="pill-val" style="color:var(--amber);">{metrics["beta"]:.2f}x</span>
          <div class="pill-bar"><div class="pill-fill" style="width:{min(max(metrics["beta"]/3,0),1)*100:.0f}%;background:var(--amber);"></div></div>
        </div>
        <div class="hero-pill">
          <span class="pill-label">Max DD</span>
          <span class="pill-val" style="color:var(--red);">{_fp(metrics["mdd"]*100)}</span>
          <div class="pill-bar"><div class="pill-fill" style="width:{min(abs(metrics['mdd'])*2,1)*100:.0f}%;background:var(--red);"></div></div>
        </div>
        <div class="hero-pill">
          <span class="pill-label">Vol (ann.)</span>
          <span class="pill-val" style="color:var(--blue);">{_fp(metrics["vol"]*100)}</span>
          <div class="pill-bar"><div class="pill-fill" style="width:{min(metrics['vol'],1)*100:.0f}%;background:var(--blue);"></div></div>
        </div>
        <div class="hero-pill">
          <span class="pill-label">Hit Ratio</span>
          <span class="pill-val" style="color:var(--green);">{_fp(metrics["hit"]*100)}</span>
          <div class="pill-bar"><div class="pill-fill" style="width:{metrics['hit']*100:.0f}%;background:var(--green);"></div></div>
        </div>
      </div>
    </div>
    """

    kpi_grid = "".join([
        _kpi("Ann. Return", _fp(metrics["ann_return"]*100), f"QQQ: {_fp(metrics['ann_bench']*100)}", "g"),
        _kpi("CAPM Alpha", _fp(metrics["jalpha"]*100), "Jensen's α (ann.)", "b"),
        _kpi("Beta vs QQQ", f"{metrics['beta']:.2f}x", f"ρ={metrics['corr']:.3f}", "a"),
        _kpi("VaR 95% (1d)", _fp(metrics["var95"]*100), f"CVaR: {_fp(metrics['cvar95']*100)}", "r"),
        _kpi("Info Ratio", f"{metrics['ir']:.3f}", f"TE: {_fp(metrics['te']*100)}", "p"),
    ])

    intelligence_html = "".join(
        f"<div class='intel-section'><div class='intel-head'>{_esc(t)}</div><div class='intel-body'>{_esc(b)}</div></div>"
        for t, b in intel
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{_esc(CFG["portfolio_name"])} — Portfolio Analytics</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {{
  --bg:#07090F;
  --panel:#0C1018;
  --card:#111520;
  --card2:#161C2A;
  --border:#1C2840;
  --border2:#243350;
  --green:#00D97E;
  --blue:#2D7DD2;
  --red:#E8304A;
  --amber:#F0A500;
  --purple:#7C5CBF;
  --cyan:#00B4D8;
  --text:#D8E2F0;
  --sub:#7A8FAD;
  --dim:#3A4A65;
  --mono:'JetBrains Mono','Courier New',monospace;
  --sans:'DM Sans',system-ui,sans-serif;
}}
*,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
html {{ scroll-behavior:smooth; }}
body {{ background:var(--bg); color:var(--text); font-family:var(--sans); font-size:13px; line-height:1.5; min-height:100vh; overflow-x:hidden; }}
::-webkit-scrollbar {{ width:4px; height:4px; }}
::-webkit-scrollbar-track {{ background:var(--panel); }}
::-webkit-scrollbar-thumb {{ background:var(--border2); border-radius:2px; }}

.topbar {{
  position:sticky; top:0; z-index:200;
  background:rgba(7,9,15,0.97);
  backdrop-filter:blur(20px);
  border-bottom:1px solid var(--border);
  display:flex; align-items:center; justify-content:space-between;
  padding:0 32px; height:52px;
}}
.logo {{ font-family:var(--mono); font-size:13px; font-weight:700; color:var(--green); letter-spacing:.3px; }}
.logo sup {{ font-size:9px; color:var(--sub); font-weight:400; margin-left:6px; }}
.topbar-nav {{ display:flex; gap:2px; }}
.topbar-nav a {{
  color:var(--sub); text-decoration:none; font-size:10.5px; font-weight:500;
  letter-spacing:.8px; text-transform:uppercase; padding:5px 12px; border-radius:4px;
  transition:all .15s; font-family:var(--mono);
}}
.topbar-nav a:hover, .topbar-nav a.active {{ color:var(--text); background:var(--card2); }}
.topbar-right {{ display:flex; align-items:center; gap:20px; }}
.live-dot {{ display:flex; align-items:center; gap:6px; font-family:var(--mono); font-size:9.5px; color:var(--green); letter-spacing:.5px; }}
.pulse {{ width:7px; height:7px; border-radius:50%; background:var(--green); animation:blink 2s infinite; box-shadow:0 0 6px var(--green); }}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:.25}} }}
.updated {{ font-family:var(--mono); font-size:10px; color:var(--sub); }}

main {{ max-width:1500px; margin:0 auto; padding:24px 32px 60px; }}
.section-label {{
  font-family:var(--mono); font-size:9px; font-weight:700; color:var(--sub);
  letter-spacing:3px; text-transform:uppercase; margin:32px 0 16px;
  display:flex; align-items:center; gap:12px;
}}
.section-label::after {{ content:''; flex:1; height:1px; background:var(--border); }}
.section-label .slnum {{ color:var(--dim); font-size:8px; }}

.row {{ display:grid; gap:16px; margin-bottom:16px; }}
.r2 {{ grid-template-columns:1fr 1fr; }}
.r3 {{ grid-template-columns:1fr 1fr 1fr; }}
.r65 {{ grid-template-columns:3fr 2fr; }}

.card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
.card-head {{ display:flex; align-items:center; justify-content:space-between; padding:12px 18px; border-bottom:1px solid var(--border); }}
.card-title {{ font-family:var(--mono); font-size:9px; font-weight:700; color:var(--sub); letter-spacing:2px; text-transform:uppercase; }}
.card-badge {{ font-family:var(--mono); font-size:8px; color:var(--dim); letter-spacing:.5px; padding:2px 8px; background:var(--card2); border-radius:3px; border:1px solid var(--border); }}
.card-body {{ padding:16px 18px; }}
.chart-wrap {{ padding:4px 6px 8px; }}

.hero {{
  display:grid; grid-template-columns:auto auto auto auto 1fr; gap:0;
  border:1px solid var(--border); border-radius:10px; background:var(--card);
  overflow:hidden; margin-bottom:16px; position:relative;
}}
.hero::before {{
  content:''; position:absolute; inset:0;
  background:radial-gradient(ellipse at 10% 50%, rgba(0,217,126,0.04) 0%, transparent 60%);
  pointer-events:none;
}}
.hero-block {{ padding:20px 28px; border-right:1px solid var(--border); }}
.hero-block:last-child {{ border-right:none; }}
.hero-label {{ font-family:var(--mono); font-size:8.5px; color:var(--sub); letter-spacing:2.5px; text-transform:uppercase; margin-bottom:8px; }}
.hero-val {{ font-family:var(--mono); font-size:28px; font-weight:700; line-height:1; letter-spacing:-1.5px; }}
.hero-val.green {{ color:var(--green); }}
.hero-val.red {{ color:var(--red); }}
.hero-val.blue {{ color:var(--blue); }}
.hero-val.amber {{ color:var(--amber); }}
.hero-sub {{ font-family:var(--mono); font-size:10px; color:var(--sub); margin-top:6px; }}
.hero-right {{ padding:16px 28px; display:flex; flex-direction:column; justify-content:center; gap:10px; }}
.hero-pill {{ display:flex; align-items:center; gap:10px; }}
.pill-label {{ font-family:var(--mono); font-size:9px; color:var(--sub); letter-spacing:1px; width:90px; text-transform:uppercase; }}
.pill-val {{ font-family:var(--mono); font-size:11px; font-weight:600; }}
.pill-bar {{ flex:1; height:3px; background:var(--border); border-radius:2px; overflow:hidden; }}
.pill-fill {{ height:100%; border-radius:2px; }}

.kpi-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:16px; }}
.kpi {{
  background:var(--card); border:1px solid var(--border); border-radius:8px;
  padding:14px 16px; position:relative; overflow:hidden;
}}
.kpi::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; }}
.kpi.g::before {{ background:var(--green); }}
.kpi.b::before {{ background:var(--blue); }}
.kpi.r::before {{ background:var(--red); }}
.kpi.a::before {{ background:var(--amber); }}
.kpi.p::before {{ background:var(--purple); }}
.kpi-label {{ font-family:var(--mono); font-size:8.5px; color:var(--sub); letter-spacing:2px; text-transform:uppercase; margin-bottom:8px; }}
.kpi-val {{ font-family:var(--mono); font-size:20px; font-weight:700; line-height:1; letter-spacing:-1px; }}
.kpi-sub {{ font-family:var(--mono); font-size:9px; color:var(--sub); margin-top:5px; }}

.bucket {{
  font-family:var(--mono); font-size:8px; padding:1px 7px; border-radius:3px;
  text-transform:uppercase; letter-spacing:.5px;
}}
.bucket.core {{ background:rgba(45,125,210,0.12); color:var(--blue); border:1px solid rgba(45,125,210,0.25); }}
.bucket.growth {{ background:rgba(0,217,126,0.10); color:var(--green); border:1px solid rgba(0,217,126,0.22); }}
.bucket.speculative {{ background:rgba(240,165,0,0.10); color:var(--amber); border:1px solid rgba(240,165,0,0.22); }}

.data-table {{ width:100%; border-collapse:collapse; font-family:var(--mono); font-size:11px; }}
.data-table th {{
  font-size:8.5px; color:var(--sub); letter-spacing:1.5px; text-transform:uppercase;
  padding:10px 12px; border-bottom:1px solid var(--border); text-align:left;
  background:var(--card2); font-weight:600; white-space:nowrap;
}}
.data-table td {{ padding:9px 12px; border-bottom:1px solid rgba(28,40,64,0.5); white-space:nowrap; }}
.data-table tr:last-child td {{ border-bottom:none; }}
.data-table tr:hover td {{ background:rgba(45,125,210,0.04); }}
.num {{ text-align:right; }}
.td-ticker {{ font-weight:700; color:var(--text); font-size:12px; }}
.td-name {{ color:var(--sub); font-size:10.5px; max-width:240px; overflow:hidden; text-overflow:ellipsis; }}

.intel-section {{ margin-bottom:20px; }}
.intel-head {{
  font-family:var(--mono); font-size:8.5px; font-weight:700; color:var(--blue);
  letter-spacing:2.5px; text-transform:uppercase; padding-bottom:8px;
  border-bottom:1px solid var(--border); margin-bottom:12px;
}}
.intel-body {{ font-size:12.5px; line-height:1.9; color:var(--text); }}

footer {{
  border-top:1px solid var(--border); padding:24px 32px; display:flex;
  align-items:center; justify-content:space-between; max-width:1500px; margin:0 auto;
}}
.footer-left,.footer-right {{ font-family:var(--mono); font-size:10px; color:var(--dim); }}

@media (max-width: 1200px) {{
  .hero {{ grid-template-columns:1fr 1fr; }}
  .kpi-grid {{ grid-template-columns:repeat(3,1fr); }}
}}
@media (max-width: 768px) {{
  main {{ padding:16px; }}
  .topbar {{ padding:0 16px; }}
  .hero {{ grid-template-columns:1fr 1fr; }}
  .kpi-grid {{ grid-template-columns:1fr 1fr; }}
  .r2,.r3,.r65 {{ grid-template-columns:1fr; }}
}}
</style>
</head>
<body>

<nav class="topbar">
  <div style="display:flex;align-items:center;gap:28px;">
    <div class="logo">Life on the Hedge Fund <sup>Trinity College Dublin</sup></div>
    <nav class="topbar-nav">
      <a href="#overview" class="active">Overview</a>
      <a href="#performance">Performance</a>
      <a href="#risk">Risk</a>
      <a href="#positions">Positions</a>
      <a href="#stress">Stress</a>
      <a href="#intelligence">Intelligence</a>
    </nav>
  </div>
  <div class="topbar-right">
    <div class="live-dot"><div class="pulse"></div>LIVE DATA</div>
    <div class="updated">Updated: {now_utc}</div>
  </div>
</nav>

<main>
  {hero}

  <div class="kpi-grid">{kpi_grid}</div>

  <div id="performance" class="section-label"><span class="slnum">01</span> Performance</div>
  <div class="card" style="margin-bottom:16px;">
    <div class="card-head">
      <span class="card-title">Portfolio vs Benchmarks — Base 100 from {CFG["inception"]}</span>
      <span class="card-badge">NAV-weighted · No browser-side fetching</span>
    </div>
    <div class="chart-wrap" id="perf"></div>
  </div>

  <div class="row r2">
    <div class="card">
      <div class="card-head"><span class="card-title">Drawdown from Peak</span><span class="card-badge">vs {CFG["benchmark"]}</span></div>
      <div class="chart-wrap" id="drawdown"></div>
    </div>
    <div class="card">
      <div class="card-head"><span class="card-title">Monthly Returns vs {CFG["benchmark"]}</span><span class="card-badge">bar + line overlay</span></div>
      <div class="chart-wrap" id="monthly_returns"></div>
    </div>
  </div>

  <div id="risk" class="section-label"><span class="slnum">02</span> Risk Analytics</div>
  <div class="row r3">
    <div class="card">
      <div class="card-head"><span class="card-title">Rolling Volatility</span></div>
      <div class="chart-wrap" id="rolling_vol"></div>
    </div>
    <div class="card">
      <div class="card-head"><span class="card-title">Rolling Beta</span></div>
      <div class="chart-wrap" id="rolling_beta"></div>
    </div>
    <div class="card">
      <div class="card-head"><span class="card-title">Rolling Sharpe</span></div>
      <div class="chart-wrap" id="rolling_sharpe"></div>
    </div>
  </div>

  <div class="card" style="margin-bottom:16px;">
    <div class="card-head">
      <span class="card-title">Complete Risk / Return Metrics</span>
      <span class="card-badge">rf={CFG["rf"]*100:.2f}% · annualised at 252 trading days</span>
    </div>
    <div class="card-body" style="padding:0;overflow-x:auto;">
      <table class="data-table">
        <thead><tr><th>Metric</th><th class="num">Value</th><th>Comment</th></tr></thead>
        <tbody>{_metrics_table(metrics, pos)}</tbody>
      </table>
    </div>
  </div>

  <div id="positions" class="section-label"><span class="slnum">03</span> Positions</div>
  <div class="row r3">
    <div class="card">
      <div class="card-head"><span class="card-title">Top Weights</span></div>
      <div class="chart-wrap" id="top_weights"></div>
    </div>
    <div class="card">
      <div class="card-head"><span class="card-title">Sector Allocation</span></div>
      <div class="chart-wrap" id="sector_alloc"></div>
    </div>
    <div class="card">
      <div class="card-head"><span class="card-title">Thematic Allocation</span></div>
      <div class="chart-wrap" id="theme_alloc"></div>
    </div>
  </div>

  <div class="row r2" style="margin-bottom:16px;">
    <div class="card">
      <div class="card-head"><span class="card-title">Position P&amp;L Attribution</span></div>
      <div class="chart-wrap" id="pnl_attr"></div>
    </div>
    <div class="card">
      <div class="card-head"><span class="card-title">Monthly Heatmap</span></div>
      <div class="chart-wrap" id="heatmap"></div>
    </div>
  </div>

  <div class="card" style="margin-bottom:16px;">
    <div class="card-head">
      <span class="card-title">Holdings — {len(pos)} Positions · NAV {_fc(metrics["current_nav"],0)}</span>
      <span class="card-badge">Buy &amp; Hold · No rebalancing since {CFG["inception"]}</span>
    </div>
    <div style="overflow-x:auto;">
      <table class="data-table">
        <thead><tr>
          <th>Ticker</th><th>Name</th><th>Sector</th><th>Theme</th><th>Risk</th>
          <th class="num">Qty</th><th class="num">Buy</th><th class="num">Price</th>
          <th class="num">Value</th><th class="num">P&amp;L</th><th class="num">Return</th>
          <th class="num">Weight</th><th class="num">Day</th><th class="num">1W</th><th class="num">1M</th><th class="num">Beta</th>
        </tr></thead>
        <tbody>{_positions_table(pos)}</tbody>
      </table>
    </div>
  </div>

  <div class="row r65">
    <div class="card">
      <div class="card-head"><span class="card-title">Daily Ledger</span></div>
      <div class="card-body" style="padding:0;overflow-x:auto;">
        <table class="data-table">
          <thead><tr><th>Date</th><th class="num">NAV</th><th class="num">Daily P&amp;L</th><th class="num">Portfolio</th><th class="num">QQQ</th><th class="num">Active</th><th class="num">Drawdown</th></tr></thead>
          <tbody>{_ledger_table(ledger)}</tbody>
        </table>
      </div>
    </div>

    <div id="intelligence" class="card">
      <div class="card-head"><span class="card-title">Portfolio Intelligence</span></div>
      <div class="card-body">{intelligence_html}</div>
    </div>
  </div>

  <div id="stress" class="section-label"><span class="slnum">04</span> Stress & Scenario</div>
  <div class="row r2">
    <div class="card">
      <div class="card-head"><span class="card-title">Stress Tests</span></div>
      <div class="chart-wrap" id="stress"></div>
      <div class="card-body" style="padding-top:0;overflow-x:auto;">
        <table class="data-table">
          <thead><tr><th>Scenario</th><th>Description</th><th class="num">P&amp;L Impact</th><th class="num">Return Impact</th><th class="num">NAV After</th></tr></thead>
          <tbody>{_stress_table(stress)}</tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-head"><span class="card-title">12M Scenario Envelope</span></div>
      <div class="chart-wrap" id="forecast"></div>
      <div class="card-body" style="padding-top:0;overflow-x:auto;">
        <table class="data-table">
          <thead><tr><th>Horizon</th><th class="num">Start NAV</th><th class="num">P05</th><th class="num">P25</th><th class="num">Median</th><th class="num">P75</th><th class="num">P95</th></tr></thead>
          <tbody>{_forecast_table(fcast_sum)}</tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="section-label"><span class="slnum">05</span> News & Monitoring</div>
  <div class="row r2">
    <div class="card">
      <div class="card-head"><span class="card-title">Sector Breakdown</span></div>
      <div class="card-body" style="padding:0;overflow-x:auto;">
        <table class="data-table">
          <thead><tr><th>Sector</th><th class="num">Weight</th><th class="num">Market Value</th><th class="num">P&amp;L</th><th class="num">N</th></tr></thead>
          <tbody>{_sector_table(structure)}</tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-head"><span class="card-title">News Flow</span></div>
      <div class="card-body" style="padding:0;overflow-x:auto;">
        <table class="data-table">
          <thead><tr><th>Ticker</th><th>Headline</th><th>Source</th><th>Published</th></tr></thead>
          <tbody>{_news_table(news)}</tbody>
        </table>
      </div>
    </div>
  </div>

</main>

<footer>
  <div class="footer-left">Python-generated static dashboard · GitHub Pages compatible · No client-side core analytics fetch</div>
  <div class="footer-right">{_esc(CFG["school"])} · {_esc(CFG["portfolio_name"])}</div>
</footer>

<script>
const CHARTS = {chart_json};
const CONFIG = {{
  displayModeBar: false,
  responsive: true,
  scrollZoom: false
}};
for (const [id, fig] of Object.entries(CHARTS)) {{
  const el = document.getElementById(id);
  if (el) {{
    Plotly.newPlot(id, fig.data, fig.layout, CONFIG);
  }}
}}
</script>
</body>
</html>"""


def build_snapshot(metrics: dict, pos: pd.DataFrame, structure: dict, ledger: pd.DataFrame,
                   stress: pd.DataFrame, fcast_sum: pd.DataFrame, news: pd.DataFrame,
                   intel: list) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": CFG,
        "metrics": {
            "current_nav": metrics["current_nav"],
            "daily_pnl": metrics["daily_pnl"],
            "daily_return": metrics["daily_return"],
            "total_pnl": metrics["total_pnl"],
            "total_return": metrics["total_return"],
            "bench_total_return": metrics["bench_total_return"],
            "alpha": metrics["alpha"],
            "ann_return": metrics["ann_return"],
            "ann_bench": metrics["ann_bench"],
            "vol": metrics["vol"],
            "bvol": metrics["bvol"],
            "ddev": metrics["ddev"],
            "bddev": metrics["bddev"],
            "sharpe": metrics["sharpe"],
            "sortino": metrics["sortino"],
            "beta": metrics["beta"],
            "corr": metrics["corr"],
            "jalpha": metrics["jalpha"],
            "te": metrics["te"],
            "ir": metrics["ir"],
            "mdd": metrics["mdd"],
            "bmdd": metrics["bmdd"],
            "calmar": metrics["calmar"],
            "var95": metrics["var95"],
            "cvar95": metrics["cvar95"],
            "skew": metrics["skew"],
            "kurt": metrics["kurt"],
            "treynor": metrics["treynor"],
            "omega": metrics["omega"],
            "upc": metrics["upc"],
            "dnc": metrics["dnc"],
            "hit": metrics["hit"],
            "sessions": metrics["sessions"],
        },
        "positions": pos.to_dict(orient="records"),
        "structure": {
            "sector": structure["sector"].to_dict(orient="records"),
            "theme": structure["theme"].to_dict(orient="records"),
            "bucket": structure["bucket"].to_dict(orient="records"),
            "hhi": structure["hhi"],
            "eff_n": structure["eff_n"],
            "top5": structure["top5"],
        },
        "ledger": ledger.to_dict(orient="records"),
        "stress": stress.to_dict(orient="records"),
        "forecast_summary": fcast_sum.to_dict(orient="records"),
        "news": news.to_dict(orient="records") if not news.empty else [],
        "intelligence": [{"title": t, "body": b} for t, b in intel],
    }


def main():
    holdings = load_holdings(ROOT / "holdings.csv")
    tickers = holdings["ticker"].tolist() + [CFG["benchmark"], CFG["bench2"]]
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    prices = download_prices(tickers, CFG["inception"], end)
    frame = build_frame(prices, holdings)
    metrics = compute_metrics(frame)
    pos = compute_positions(frame, holdings)
    structure = compute_structure(pos)
    ledger = build_ledger(frame)
    heatmap = build_heatmap(metrics["mp"])
    stress = build_stress(pos, metrics)
    fcast_sum, fcast_paths = build_forecast(frame)
    news = build_news(holdings)
    intel = build_intelligence(metrics, pos, structure, stress)
    charts = make_charts(frame, metrics, pos, structure, heatmap, stress, fcast_paths)

    html = generate_html(
        holdings, frame, metrics, pos, structure, ledger,
        heatmap, stress, fcast_sum, fcast_paths, news, intel, charts
    )
    OUT_HTML.write_text(html, encoding="utf-8", newline="\n")

    snapshot = build_snapshot(metrics, pos, structure, ledger, stress, fcast_sum, news, intel)
    OUT_JSON.write_text(_dumps(snapshot), encoding="utf-8", newline="\n")

    print(f"Generated {OUT_HTML}")
    print(f"Generated {OUT_JSON}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Build failed: {exc}")
        traceback.print_exc()
        raise
