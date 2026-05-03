"""마켓 타이밍 레이어: KOSPI 지수의 이동평균 기반 약세장 회피 모듈.

리밸런싱 시점에 KOSPI 지수가 N일 이동평균 아래에 있으면 현금 비중을
높여 약세장 피해를 줄입니다.

사용법:
    from market_timing import check_market_timing, MarketTimingDecision
    decision = check_market_timing(date_str, enabled=True, ma_days=200)
    if decision.in_market:
        # 정상 투자
    else:
        # 현금 보유
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarketTimingDecision:
    """마켓 타이밍 판단 결과."""
    enabled: bool
    in_market: bool          # True: 지수가 MA 위 → 정상 투자
    kospi_close: float | None
    ma_value: float | None
    ma_days: int
    reason: str
    investment_ratio_multiplier: float  # 1.0: 정상, 0.0: 전액 현금


def check_market_timing(
    date_str: str,          # YYYY-MM-DD
    enabled: bool = True,
    ma_days: int = 200,
    cash_ratio_when_bearish: float = 1.0,  # 약세장 시 현금 비중 (1.0=전액 현금)
) -> MarketTimingDecision:
    """KOSPI 지수가 ma_days 이동평균 위/아래인지 확인합니다.

    Args:
        date_str: 확인 날짜 (YYYY-MM-DD)
        enabled: False이면 항상 in_market=True(정상 투자)
        ma_days: 이동평균 기간 (기본 200일)
        cash_ratio_when_bearish: 약세장 시 현금 비중 (0.0~1.0)
            1.0 = 전액 현금, 0.5 = 절반만 현금

    Returns:
        MarketTimingDecision
    """
    if not enabled:
        return MarketTimingDecision(
            enabled=False,
            in_market=True,
            kospi_close=None,
            ma_value=None,
            ma_days=ma_days,
            reason="disabled",
            investment_ratio_multiplier=1.0,
        )

    try:
        import pandas as pd
        from pykrx import stock as pykrx_stock

        # MA 계산에 필요한 충분한 기간 조회 (ma_days + 여유 60일)
        end_dt = pd.to_datetime(date_str)
        start_dt = end_dt - pd.DateOffset(days=ma_days + 60)
        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_dt.strftime("%Y%m%d")

        # KOSPI 지수 OHLCV 조회 (티커: '1001' = KOSPI)
        df = pykrx_stock.get_index_ohlcv(start_str, end_str, "1001")

        if df is None or df.empty:
            # 데이터 없으면 보수적으로 in_market=True (투자 허용)
            return MarketTimingDecision(
                enabled=True,
                in_market=True,
                kospi_close=None,
                ma_value=None,
                ma_days=ma_days,
                reason="data_unavailable(fallback=in_market)",
                investment_ratio_multiplier=1.0,
            )

        close_col = "종가" if "종가" in df.columns else df.columns[3]
        closes = df[close_col].astype(float)

        if len(closes) < 2:
            return MarketTimingDecision(
                enabled=True,
                in_market=True,
                kospi_close=None,
                ma_value=None,
                ma_days=ma_days,
                reason="insufficient_data(fallback=in_market)",
                investment_ratio_multiplier=1.0,
            )

        # 사용 가능한 데이터로 MA 계산 (데이터가 ma_days보다 짧으면 가용 기간으로 계산)
        actual_days = min(ma_days, len(closes))
        ma_value = float(closes.iloc[-actual_days:].mean())
        kospi_close = float(closes.iloc[-1])
        in_market = kospi_close >= ma_value

        if in_market:
            multiplier = 1.0
            reason = f"above_MA{actual_days}({kospi_close:,.0f}>={ma_value:,.0f})"
        else:
            multiplier = 1.0 - cash_ratio_when_bearish
            reason = f"below_MA{actual_days}({kospi_close:,.0f}<{ma_value:,.0f})"

        return MarketTimingDecision(
            enabled=True,
            in_market=in_market,
            kospi_close=kospi_close,
            ma_value=ma_value,
            ma_days=actual_days,
            reason=reason,
            investment_ratio_multiplier=multiplier,
        )

    except Exception as e:
        return MarketTimingDecision(
            enabled=True,
            in_market=True,
            kospi_close=None,
            ma_value=None,
            ma_days=ma_days,
            reason=f"error({e})(fallback=in_market)",
            investment_ratio_multiplier=1.0,
        )
