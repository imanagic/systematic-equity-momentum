from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from src.config import BacktestConfig, DEFAULT_TICKERS
from src.data_loader import load_adj_close
from src.signals import momentum_12_1
from src.backtest import run_long_short_backtest
from src.metrics import summary_table


def main() -> None:
    print("RUN_BACKTEST: starting")

    cfg = BacktestConfig(tickers=DEFAULT_TICKERS)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    print("RUN_BACKTEST: downloading/loading data")
    prices = load_adj_close(
        tickers=cfg.tickers,
        benchmark=cfg.benchmark,
        start=cfg.start,
        end=cfg.end,
        cache_path="data/adj_close.csv",
    )
    print(f"RUN_BACKTEST: prices loaded with shape={prices.shape}")

    print("RUN_BACKTEST: computing signal")
    signal = momentum_12_1(prices, lookback=cfg.lookback_days, skip=cfg.skip_days)

    print("RUN_BACKTEST: running backtest")
    results = run_long_short_backtest(
        prices=prices,
        signal=signal,
        benchmark=cfg.benchmark,
        quantile=cfg.quantile,
        gross_exposure=cfg.gross_exposure,
        cost_per_dollar=cfg.cost_per_dollar,
    )

    port = results["daily_returns"]
    bench = results["benchmark_returns"]

    print("RUN_BACKTEST: computing metrics")
    metrics_df = summary_table(port, bench)

    print("\n=== Strategy Metrics ===")
    print(metrics_df.round(4).to_string(index=False))

    metrics_path = Path("reports/metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    strat_equity = (1 + port).cumprod()
    bench_equity = (1 + bench.reindex(port.index).fillna(0.0)).cumprod()

    print("RUN_BACKTEST: saving plot")
    plt.figure()
    plt.plot(strat_equity.index, strat_equity.values, label="Strategy")
    plt.plot(bench_equity.index, bench_equity.values, label="SPY")
    plt.title("Equity Curve: Strategy vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()

    plot_path = Path("reports/equity_curve.png")
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print(f"RUN_BACKTEST: saved {metrics_path}")
    print(f"RUN_BACKTEST: saved {plot_path}")
    print("RUN_BACKTEST: done")


if __name__ == "__main__":
    main()
