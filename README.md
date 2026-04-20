# Life on the Hedge Fund

Institutional recruiter-facing portfolio dashboard for a fictional academic portfolio built at Trinity College Dublin.

## Positioning

- Portfolio name: Life on the Hedge Fund
- Institution: Trinity College Dublin
- Context: academic portfolio analytics project
- Universe: concentrated US equity
- Initial AUM: around 50,000 USD
- Benchmark: QQQ
- Secondary benchmark: SPY
- Rebalancing: none since inception
- Horizon: 15 years
- Style: high-conviction, high-beta, thematic growth

## Architecture

Python is the single source of truth.

`build_dashboard.py`:
- loads `holdings.csv`
- fetches market data with `yfinance`
- computes portfolio and benchmark analytics in Python
- generates static `docs/index.html`
- writes `data/dashboard_snapshot.json`

There is no browser-side fetching for core analytics.

## Repository structure

```text
.
├── build_dashboard.py
├── holdings.csv
├── requirements.txt
├── README.md
└── .github/
    └── workflows/
        └── update-dashboard.yml
