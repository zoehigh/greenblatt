from __future__ import annotations

from dataclasses import dataclass
from defaults import DEFAULT_REBALANCE_MONTHS

try:
    from dotenv import find_dotenv, load_dotenv
    from utils.env import env_get
except ImportError:  # pragma: no cover
    find_dotenv = None  # type: ignore[assignment]
    load_dotenv = None  # type: ignore[assignment]
    env_get = None  # type: ignore[assignment]


@dataclass(slots=True)
class StrategyConfig:
    """백테스트와 실거래에서 공통으로 사용하는 전략 파라미터."""

    # --- 종목 선정 ---
    investment_ratio: float = 0.95
    num_stocks: int = 40
    rebalance_days: int | None = None
    rebalance_months: int = DEFAULT_REBALANCE_MONTHS
    strategy_mode: str = "mixed"
    mixed_filter_profile: str = "large_cap"
    kosdaq_target_ratio: float | None = None
    momentum_enabled: bool = True
    momentum_months: int = 3
    momentum_weight: float = 0.60
    momentum_filter_enabled: bool = True
    large_cap_min_mcap: float | None = None
    sector_max_stocks: int | None = None
    market_timing_enabled: bool = False
    market_timing_ma_days: int = 200
    market_timing_cash_ratio: float = 1.0
    fundamental_source: str = "pykrx"
    # --- 비용 ---
    commission_fee_rate: float = 0.00015
    tax_rate: float = 0.002
    # --- 변동성 타게팅 ---
    vol_target_enabled: bool = True
    vol_target_sigma: float = 0.28
    vol_target_lookback: int = 60
    vol_target_min_ratio: float = 0.65
    # --- 자본 제약 선택 ---
    capital_constrained_selection_enabled: bool = True
    capital_constrained_min_stocks: int = 20
    capital_constrained_max_stocks: int = 40


@dataclass(slots=True)
class BacktestConfig(StrategyConfig):
    """백테스트 전용 파라미터. StrategyConfig를 상속하여 전략·비용 파라미터를 공유한다."""

    start_date: str = "2017-05-01"
    end_date: str = "2025-04-30"
    initial_capital: int = 10_000_000
    sell_losers_enabled: bool = True
    sell_losers_hold_days: int = 365
    slippage_bps: int = 30
    cache_dir: str = "results/cache"
    timing_enabled: bool = True
    fundamental_cache_format: str = "parquet"
    fundamental_cache_max_entries: int = 16
    use_open_price: bool = False

    @classmethod
    def from_env(cls) -> "BacktestConfig":
        """환경변수 및 .env 파일에서 BacktestConfig를 생성한다."""
        if load_dotenv is not None:
            dotenv_path = find_dotenv(usecwd=True) if find_dotenv else ""
            if dotenv_path:
                load_dotenv(dotenv_path=dotenv_path, override=False)
            else:
                load_dotenv(override=False)

        # rebalance_days 파싱
        reb_days_env = env_get("REBALANCE_DAYS", fallback_keys=["LIVE_REBALANCE_DAYS"])
        rebalance_days_val: int | None = None
        if reb_days_env not in (None, ""):
            try:
                parsed = int(reb_days_env)
                if parsed > 0:
                    rebalance_days_val = parsed
            except Exception:
                pass

        reb_env = env_get(
            "REBALANCE_MONTHS",
            fallback_keys=["LIVE_REBALANCE_MONTHS"],
            default=str(DEFAULT_REBALANCE_MONTHS),
        )
        try:
            rebalance_months_val = int(reb_env)
        except Exception:
            rebalance_months_val = int(str(DEFAULT_REBALANCE_MONTHS))

        num_stocks_val = int(
            env_get("NUM_STOCKS", fallback_keys=["BACKTEST_NUM_STOCKS", "LIVE_NUM_STOCKS"], default="40")
        )
        cap_max = int(
            env_get(
                "CAPITAL_CONSTRAINED_MAX_STOCKS",
                fallback_keys=["LIVE_CAPITAL_CONSTRAINED_MAX_STOCKS", "LIVE_NUM_STOCKS"],
                default=str(num_stocks_val),
            )
        )

        # sell_losers_hold_days 파싱
        hold_days = 365
        env_hold = env_get("SELL_LOSERS_HOLD_DAYS")
        if env_hold not in (None, ""):
            try:
                parsed_hold = int(env_hold)
                if parsed_hold > 0:
                    hold_days = parsed_hold
            except Exception:
                pass
        else:
            env_cycles = env_get("SELL_LOSERS_HOLD_REBALANCE_CYCLES")
            if env_cycles not in (None, ""):
                try:
                    cycles = int(env_cycles)
                    if cycles > 0:
                        base_days = rebalance_days_val if rebalance_days_val else rebalance_months_val * 30
                        hold_days = max(1, cycles * base_days)
                except Exception:
                    pass

        lcap_env = env_get(
            "LARGE_CAP_MIN_MCAP",
            fallback_keys=["BACKTEST_LARGE_CAP_MIN_MCAP", "LIVE_LARGE_CAP_MIN_MCAP"],
        )

        return cls(
            # StrategyConfig 공통 필드
            investment_ratio=float(
                env_get(
                    "INVESTMENT_RATIO",
                    fallback_keys=["BACKTEST_INVESTMENT_RATIO", "LIVE_INVESTMENT_RATIO"],
                    default="0.95",
                )
            ),
            num_stocks=num_stocks_val,
            rebalance_days=rebalance_days_val,
            rebalance_months=rebalance_months_val,
            strategy_mode=env_get(
                "STRATEGY_MODE",
                fallback_keys=["BACKTEST_STRATEGY_MODE", "LIVE_STRATEGY_MODE"],
                default="mixed",
            ),
            mixed_filter_profile=env_get(
                "MIXED_FILTER_PROFILE",
                fallback_keys=["BACKTEST_MIX_PROFILE", "LIVE_MIXED_FILTER_PROFILE"],
                default="large_cap",
            ),
            kosdaq_target_ratio=None,
            momentum_enabled=str(
                env_get(
                    "MOMENTUM_ENABLED",
                    fallback_keys=["BACKTEST_MOMENTUM_ENABLED", "LIVE_MOMENTUM_ENABLED"],
                    default="true",
                )
            ).lower() in {"1", "true", "yes", "y"},
            momentum_months=int(
                env_get(
                    "MOMENTUM_MONTHS",
                    fallback_keys=["BACKTEST_MOMENTUM_MONTHS", "LIVE_MOMENTUM_MONTHS"],
                    default="6",
                )
            ),
            momentum_weight=float(
                env_get(
                    "MOMENTUM_WEIGHT",
                    fallback_keys=["BACKTEST_MOMENTUM_WEIGHT", "LIVE_MOMENTUM_WEIGHT"],
                    default="0.50",
                )
            ),
            momentum_filter_enabled=str(
                env_get(
                    "MOMENTUM_FILTER_ENABLED",
                    fallback_keys=["BACKTEST_MOMENTUM_FILTER_ENABLED", "LIVE_MOMENTUM_FILTER_ENABLED"],
                    default="true",
                )
            ).lower() in {"1", "true", "yes", "y"},
            large_cap_min_mcap=float(lcap_env) if lcap_env else None,
            sector_max_stocks=int(env_get("SECTOR_MAX_STOCKS", default="0")) or None,
            market_timing_enabled=str(
                env_get("MARKET_TIMING_ENABLED", default="false")
            ).lower() in {"1", "true", "yes", "y"},
            market_timing_ma_days=int(env_get("MARKET_TIMING_MA_DAYS", default="200")),
            market_timing_cash_ratio=float(env_get("MARKET_TIMING_CASH_RATIO", default="1.0")),
            fundamental_source=env_get(
                "FUNDAMENTAL_SOURCE",
                fallback_keys=["BACKTEST_FUNDAMENTAL_SOURCE", "LIVE_FUNDAMENTAL_SOURCE"],
                default="pykrx",
            ).strip().lower(),
            commission_fee_rate=float(
                env_get(
                    "COMMISSION_FEE_RATE",
                    fallback_keys=["BACKTEST_COMMISSION_FEE_RATE", "LIVE_COMMISSION_FEE_RATE"],
                    default="0.0015",
                )
            ),
            tax_rate=float(
                env_get("TAX_RATE", fallback_keys=["BACKTEST_TAX_RATE", "LIVE_TAX_RATE"], default="0.002")
            ),
            vol_target_enabled=str(
                env_get(
                    "VOL_TARGET_ENABLED",
                    fallback_keys=["BACKTEST_VOL_TARGET_ENABLED", "LIVE_VOL_TARGET_ENABLED"],
                    default="true",
                )
            ).lower() in {"1", "true", "yes", "y"},
            vol_target_sigma=float(
                env_get(
                    "VOL_TARGET_SIGMA",
                    fallback_keys=["BACKTEST_VOL_TARGET_SIGMA", "LIVE_VOL_TARGET_SIGMA"],
                    default="0.28",
                )
            ),
            vol_target_lookback=int(
                env_get(
                    "VOL_TARGET_LOOKBACK",
                    fallback_keys=["BACKTEST_VOL_TARGET_LOOKBACK", "LIVE_VOL_TARGET_LOOKBACK"],
                    default="60",
                )
            ),
            vol_target_min_ratio=float(
                env_get(
                    "VOL_TARGET_MIN_RATIO",
                    fallback_keys=["BACKTEST_VOL_TARGET_MIN_RATIO", "LIVE_VOL_TARGET_MIN_RATIO"],
                    default="0.65",
                )
            ),
            capital_constrained_selection_enabled=str(
                env_get(
                    "CAPITAL_CONSTRAINED_SELECTION_ENABLED",
                    fallback_keys=["LIVE_CAPITAL_CONSTRAINED_SELECTION_ENABLED"],
                    default="true",
                )
            ).lower() in {"1", "true", "yes", "y"},
            capital_constrained_min_stocks=int(
                env_get(
                    "CAPITAL_CONSTRAINED_MIN_STOCKS",
                    fallback_keys=["LIVE_CAPITAL_CONSTRAINED_MIN_STOCKS"],
                    default="20",
                )
            ),
            capital_constrained_max_stocks=cap_max,
            # BacktestConfig 전용 필드
            start_date=env_get("START_DATE", fallback_keys=["BACKTEST_START_DATE"], default="2017-05-01"),
            end_date=env_get("END_DATE", fallback_keys=["BACKTEST_END_DATE"], default="2025-04-30"),
            initial_capital=int(
                env_get("INITIAL_CAPITAL", fallback_keys=["BACKTEST_INITIAL_CAPITAL"], default="10000000")
            ),
            sell_losers_enabled=str(
                env_get(
                    "SELL_LOSERS_ENABLED",
                    fallback_keys=["BACKTEST_SELL_LOSERS_ENABLED"],
                    default="true",
                )
            ).lower() in {"1", "true", "yes", "y"},
            sell_losers_hold_days=hold_days,
            slippage_bps=int(
                env_get(
                    "SLIPPAGE_BPS",
                    fallback_keys=["BACKTEST_SLIPPAGE_BPS", "LIVE_ORDER_PRICE_OFFSET_BPS"],
                    default="30",
                )
            ),
            cache_dir=env_get("CACHE_DIR", fallback_keys=["BACKTEST_CACHE_DIR"], default="results/cache"),
            timing_enabled=str(
                env_get("TIMING_ENABLED", fallback_keys=["BACKTEST_TIMING_ENABLED"], default="true")
            ).lower() in {"1", "true", "yes", "y"},
            fundamental_cache_format=env_get(
                "FUNDAMENTAL_CACHE_FORMAT",
                fallback_keys=["BACKTEST_FUNDAMENTAL_CACHE_FORMAT"],
                default="parquet",
            ),
            fundamental_cache_max_entries=int(
                env_get(
                    "FUNDAMENTAL_CACHE_MAX_ENTRIES",
                    fallback_keys=["BACKTEST_FUNDAMENTAL_CACHE_MAX_ENTRIES"],
                    default="16",
                )
            ),
            use_open_price=str(
                env_get("USE_OPEN_PRICE", fallback_keys=["BACKTEST_USE_OPEN_PRICE"], default="false")
            ).lower() in {"1", "true", "yes", "y"},
        )
