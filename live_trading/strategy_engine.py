from __future__ import annotations

import pandas as pd

from stock_selector import KoreaStockSelector
from live_trading.strategy_config import StrategyConfig


class LiveSignalEngine:
    """실거래 신호 생성을 위한 전략 엔진.

    내부적으로 백테스트의 종목선정 로직을 재사용하되,
    실거래 엔트리에서 백테스트 기간/초기자본 개념을 노출하지 않는다.
    """

    def __init__(self, strategy_config: StrategyConfig) -> None:
        self.strategy_config = strategy_config
        self._selector = KoreaStockSelector(
            num_stocks=strategy_config.num_stocks,
            strategy_mode=strategy_config.strategy_mode,
            mixed_filter_profile=strategy_config.mixed_filter_profile,
            kosdaq_target_ratio=strategy_config.kosdaq_target_ratio,
            momentum_enabled=strategy_config.momentum_enabled,
            momentum_months=strategy_config.momentum_months,
            momentum_weight=strategy_config.momentum_weight,
            momentum_filter_enabled=strategy_config.momentum_filter_enabled,
            large_cap_min_mcap=strategy_config.large_cap_min_mcap,
            sector_max_stocks=strategy_config.sector_max_stocks,
            fundamental_source=strategy_config.fundamental_source,
        )

    @property
    def strategy_mode(self) -> str:
        return self._selector.strategy_mode

    def nearest_trading_date(self, signal_date: str) -> str:
        return self._selector.nearest_trading_date(signal_date)

    def previous_trading_date(self, date_str: str) -> str:
        return self._selector.previous_trading_date(date_str)

    def select_stocks(self, trading_date: str) -> pd.DataFrame:
        return self._selector.select_stocks(trading_date)
