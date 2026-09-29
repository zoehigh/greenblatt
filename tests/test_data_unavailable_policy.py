import os
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

# 테스트 실행 시 tests/가 작업 디렉터리로 설정되는 환경을 고려하여
# 프로젝트 루트(한 단계 위)를 PYTHONPATH에 추가합니다.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import greenblatt_korea_full_backtest as gbf
from greenblatt_korea_full_backtest import KoreaStockBacktest
from stock_selector import ScreenStatus


class DataUnavailableSelector:
    """매 호출마다 DATA_UNAVAILABLE 상태를 반환하는 더미 selector (네트워크 없음)."""

    def __init__(self, *args, **kwargs):
        self.last_screen_status = ScreenStatus.OK

    def nearest_trading_date(self, d):
        return d.replace('-', '')

    def previous_trading_date(self, d):
        return d.replace('-', '')

    def select_stocks(self, d):
        import pandas as pd
        self.last_screen_status = ScreenStatus.DATA_UNAVAILABLE
        return pd.DataFrame()

    def persist_caches(self):
        return None

    def get_market_tickers(self, d):
        return []


class MixedStatusSelector:
    """첫 호출은 DATA_UNAVAILABLE, 이후 호출은 NO_ELIGIBLE_STOCKS (hold_and_mark 테스트용)."""

    def __init__(self, *args, **kwargs):
        self.last_screen_status = ScreenStatus.OK
        self.call_count = 0

    def nearest_trading_date(self, d):
        return d.replace('-', '')

    def previous_trading_date(self, d):
        return d.replace('-', '')

    def select_stocks(self, d):
        import pandas as pd
        self.call_count += 1
        self.last_screen_status = (
            ScreenStatus.DATA_UNAVAILABLE
            if self.call_count == 1
            else ScreenStatus.NO_ELIGIBLE_STOCKS
        )
        return pd.DataFrame()

    def persist_caches(self):
        return None

    def get_market_tickers(self, d):
        return []


def _fake_get_market_ohlcv(start, end, ticker):
    """네트워크 없이 일별 종가를 반환하는 pykrx 대역."""
    import pandas as pd
    idx = pd.date_range('2020-01-01', periods=5, freq='B')
    return pd.DataFrame({'종가': [100.0, 101.0, 99.0, 102.0, 100.0]}, index=idx)


def _make_backtest(monkeypatch, *, policy='strict'):
    monkeypatch.setattr(gbf, 'KoreaStockSelector', DataUnavailableSelector)
    monkeypatch.setattr(gbf, 'LIBRARIES_AVAILABLE', True)
    return KoreaStockBacktest(
        start_date='2020-01-01',
        end_date='2020-01-31',
        sell_losers_enabled=True,
        data_unavailable_policy=policy,
    )


# ---------------------------------------------------------------------------
# strict 정책: 첫 DATA_UNAVAILABLE 발생 즉시 RuntimeError
# ---------------------------------------------------------------------------

def test_strict_raises_on_first_data_unavailable(monkeypatch):
    bt = _make_backtest(monkeypatch, policy='strict')
    with pytest.raises(RuntimeError, match=r"DATA_UNAVAILABLE \(strict\)"):
        bt.run_backtest()


def test_strict_raise_includes_date_and_status(monkeypatch):
    bt = _make_backtest(monkeypatch, policy='strict')
    with pytest.raises(RuntimeError) as excinfo:
        bt.run_backtest()
    msg = str(excinfo.value)
    assert '2020-01-01' in msg
    assert ScreenStatus.DATA_UNAVAILABLE.value in msg


def test_default_policy_is_strict():
    bt = KoreaStockBacktest(rebalance_months=None)
    assert bt.data_unavailable_policy == 'strict'


# ---------------------------------------------------------------------------
# hold_and_mark: DATA_UNAVAILABLE 발생 시 보유 유지 마크 + degraded 결과
# ---------------------------------------------------------------------------

def test_hold_and_mark_records_degraded_results(monkeypatch):
    monkeypatch.setattr(gbf, 'KoreaStockSelector', MixedStatusSelector)
    monkeypatch.setattr(gbf, 'LIBRARIES_AVAILABLE', True)
    monkeypatch.setattr(gbf.stock, 'get_market_ohlcv', _fake_get_market_ohlcv)

    bt = KoreaStockBacktest(
        start_date='2020-01-01',
        end_date='2020-03-31',
        rebalance_months=1,
        rebalance_days=None,
        sell_losers_enabled=True,
        data_unavailable_policy='hold_and_mark',
    )
    bt.portfolio = {
        '0001': {
            'ticker': '0001',
            'shares': 10,
            'buy_price': 100.0,
            'buy_date': '2020-01-01',
            'current_price': 90.0,
        }
    }
    bt.cash = 1000.0

    res = bt.run_backtest()
    assert res is not None
    assert res['degraded'] is True
    assert res['num_data_missing_days'] == 1
    assert res['data_missing_dates'] == ['2020-01-01']


# ---------------------------------------------------------------------------
# BacktestConfig: DATA_UNAVAILABLE_POLICY 파싱/검증
# ---------------------------------------------------------------------------

def test_backtest_config_policy_from_env(monkeypatch):
    monkeypatch.setenv('DATA_UNAVAILABLE_POLICY', 'hold_and_mark')
    from live_trading.strategy_config import BacktestConfig
    cfg = BacktestConfig.from_env()
    assert cfg.data_unavailable_policy == 'hold_and_mark'


def test_backtest_config_invalid_policy_raises(monkeypatch):
    monkeypatch.setenv('DATA_UNAVAILABLE_POLICY', 'bogus')
    from live_trading.strategy_config import BacktestConfig
    with pytest.raises(ValueError, match='DATA_UNAVAILABLE_POLICY'):
        BacktestConfig.from_env()


# ---------------------------------------------------------------------------
# gap-aware Sharpe: date gap > 4일인 수익률은 일별 변동성 산정에서 제외
# ---------------------------------------------------------------------------

def test_gap_aware_sharpe_excludes_gap_spanning_return(monkeypatch):
    bt = _make_backtest(monkeypatch, policy='strict')
    bt.initial_capital = 100.0

    # 01-03 → 01-13 사이에 10일 gap. 99→50 수익률(약 -49.5%)이 gap-spanning으로 제외됨.
    dates = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-13', '2020-01-14', '2020-01-15']
    values = [100.0, 101.0, 99.0, 50.0, 51.0, 52.0]
    bt.portfolio_history = [
        {'date': d, 'portfolio_value': v} for d, v in zip(dates, values)
    ]

    res = bt.calculate_performance()
    df = res['portfolio_df']
    years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    cagr = (values[-1] / bt.initial_capital) ** (1 / years) - 1

    # gap-spanning 수익률(99→50)만 제외한 일별 수익률로 기대 샤프 계산 (pandas std = ddof=1)
    expected_rets = np.array([
        values[1] / values[0] - 1,
        values[2] / values[1] - 1,
        values[4] / values[3] - 1,
        values[5] / values[4] - 1,
    ])
    expected_vol = float(np.std(expected_rets, ddof=1) * np.sqrt(252))
    expected_sharpe = (cagr - 0.035) / expected_vol if expected_vol > 0 else 0

    assert res['sharpe_ratio'] == pytest.approx(expected_sharpe)


def test_gap_aware_sharpe_keeps_sub_4day_gaps(monkeypatch):
    # 동일 수익률 열이지만 3일 gap(주말 포함)은 4일 이하로 유지되어 제외되지 않는다.
    bt = _make_backtest(monkeypatch, policy='strict')
    bt.initial_capital = 100.0

    dates = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-06', '2020-01-07', '2020-01-08']
    values = [100.0, 101.0, 99.0, 50.0, 51.0, 52.0]
    bt.portfolio_history = [
        {'date': d, 'portfolio_value': v} for d, v in zip(dates, values)
    ]

    res = bt.calculate_performance()
    df = res['portfolio_df']
    years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    cagr = (values[-1] / bt.initial_capital) ** (1 / years) - 1

    # 3일 gap은 제외되지 않으므로 5개 수익률 모두 사용
    expected_rets = np.array([
        values[1] / values[0] - 1,
        values[2] / values[1] - 1,
        values[3] / values[2] - 1,
        values[4] / values[3] - 1,
        values[5] / values[4] - 1,
    ])
    expected_vol = float(np.std(expected_rets, ddof=1) * np.sqrt(252))
    expected_sharpe = (cagr - 0.035) / expected_vol if expected_vol > 0 else 0

    assert res['sharpe_ratio'] == pytest.approx(expected_sharpe)