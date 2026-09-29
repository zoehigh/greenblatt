"""변동성 타게팅(Volatility Targeting) 모듈

포트폴리오의 실현 변동성을 목표 변동성에 맞게 투자비율을 동적으로 조정한다.

공식:
    effective_ratio = min(base_ratio, base_ratio * (σ_target / σ_realized))

동작 방식:
- 실현 변동성(σ_realized)이 목표 변동성(σ_target)보다 낮으면: base_ratio 그대로 유지 (풀 투자)
- 실현 변동성이 목표보다 높으면: 비율을 낮춰 리스크 축소
- 데이터 부족/조회 실패 시: base_ratio 그대로 유지 (fail-open)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# 일별 수익률 산정에서 제외할 date gap 임계값(일). 결측 구간(데이터 부재, 리밸런싱
# 건너뜀)이 연속 거래처럼 변동성을 부풀리는 착시를 방지하기 위해 이보다 큰 간격의
# 수익률은 실현 변동성 계산에서 제외한다.
_GAP_EXCLUSION_DAYS = 4


@dataclass(slots=True)
class VolTargetDecision:
    enabled: bool
    base_ratio: float
    effective_ratio: float
    sigma_target: float
    sigma_realized: float | None
    multiplier: float
    lookback_days: int
    data_points: int
    reason: str


def compute_vol_target_ratio(
    portfolio_history: list[dict],
    base_ratio: float,
    *,
    enabled: bool,
    sigma_target: float = 0.28,
    lookback_days: int = 60,
    min_ratio: float = 0.65,
) -> VolTargetDecision:
    """포트폴리오 히스토리 기반 변동성 타게팅 유효 투자비율을 계산한다.

    Parameters
    ----------
    portfolio_history:
        {'date': str, 'portfolio_value': float, ...} 딕셔너리 리스트.
        백테스트 `self.portfolio_history` 를 그대로 넘기면 된다.
    base_ratio:
        변동성 조건이 충족될 때 사용하는 최대 투자비율 (예: 0.95).
    enabled:
        False이면 즉시 base_ratio 반환 (기능 비활성화).
    sigma_target:
        연환산 목표 변동성 (예: 0.28 = 28%). 이 수준 이하면 풀 투자 유지.
    lookback_days:
        일별 수익률 계산에 사용할 거래일 수. 기본 60일(약 3개월).
    min_ratio:
        투자비율의 하한선. 극단적 하락 시에도 이 이하로 내려가지 않는다.
    """
    safe_base = float(max(0.0, min(1.0, base_ratio)))
    safe_sigma_target = float(max(0.01, sigma_target))
    safe_lookback = max(5, int(lookback_days))
    safe_min_ratio = float(max(0.0, min(safe_base, min_ratio)))

    if not enabled:
        return VolTargetDecision(
            enabled=False,
            base_ratio=safe_base,
            effective_ratio=safe_base,
            sigma_target=safe_sigma_target,
            sigma_realized=None,
            multiplier=1.0,
            lookback_days=safe_lookback,
            data_points=0,
            reason="disabled",
        )

    # 히스토리에서 일별 수익률 추출
    if not portfolio_history or len(portfolio_history) < safe_lookback + 1:
        return VolTargetDecision(
            enabled=True,
            base_ratio=safe_base,
            effective_ratio=safe_base,
            sigma_target=safe_sigma_target,
            sigma_realized=None,
            multiplier=1.0,
            lookback_days=safe_lookback,
            data_points=len(portfolio_history),
            reason="insufficient_history",
        )

    # 최근 lookback_days 기간의 일별 수익률
    recent = portfolio_history[-(safe_lookback + 1):]
    rows = [r for r in recent if r.get("portfolio_value") is not None]

    if len(rows) < safe_lookback + 1:
        return VolTargetDecision(
            enabled=True,
            base_ratio=safe_base,
            effective_ratio=safe_base,
            sigma_target=safe_sigma_target,
            sigma_realized=None,
            multiplier=1.0,
            lookback_days=safe_lookback,
            data_points=len(rows),
            reason="insufficient_history",
        )

    values = [r["portfolio_value"] for r in rows]
    arr = np.array(values, dtype=float)
    daily_returns = arr[1:] / arr[:-1] - 1.0

    # gap-spanning 수익률(date gap > 4일)은 일별 변동성 산정에서 제외한다.
    # 백테스트/실전 히스토리의 결측 구간(데이터 부재, 리밸런싱 건너뜀)이 연속 거래처럼
    # 변동성을 부풀리는 착시를 방지한다. 날짜 필드가 없거나 파싱 불가하면 기존 방식대로
    # 계산한다 (defense-in-depth, fail-open).
    dates = [r.get("date") for r in rows]
    if all(dates):
        try:
            gap_days = np.array(
                [
                    (pd.Timestamp(b) - pd.Timestamp(a)).days
                    for a, b in zip(dates[:-1], dates[1:])
                ]
            )
            daily_returns = daily_returns[gap_days <= _GAP_EXCLUSION_DAYS]
        except Exception:
            pass

    if len(daily_returns) == 0:
        return VolTargetDecision(
            enabled=True,
            base_ratio=safe_base,
            effective_ratio=safe_base,
            sigma_target=safe_sigma_target,
            sigma_realized=None,
            multiplier=1.0,
            lookback_days=safe_lookback,
            data_points=len(values) - 1,
            reason="insufficient_history",
        )

    sigma_daily = float(np.std(daily_returns, ddof=1))
    sigma_realized = sigma_daily * np.sqrt(252)  # 연환산

    if sigma_realized <= 0:
        return VolTargetDecision(
            enabled=True,
            base_ratio=safe_base,
            effective_ratio=safe_base,
            sigma_target=safe_sigma_target,
            sigma_realized=float(sigma_realized),
            multiplier=1.0,
            lookback_days=safe_lookback,
            data_points=len(daily_returns),
            reason="zero_volatility",
        )

    # 목표 변동성 기반 투자비율 조정
    # sigma_realized <= sigma_target 이면 multiplier >= 1.0 → cap at 1.0 (풀 투자 유지)
    raw_multiplier = safe_sigma_target / sigma_realized
    multiplier = min(1.0, raw_multiplier)
    effective = max(safe_min_ratio, safe_base * multiplier)

    return VolTargetDecision(
        enabled=True,
        base_ratio=safe_base,
        effective_ratio=float(effective),
        sigma_target=safe_sigma_target,
        sigma_realized=float(sigma_realized),
        multiplier=float(multiplier),
        lookback_days=safe_lookback,
        data_points=len(daily_returns),
        reason="ok",
    )


def warmup_vol_history(
    tickers: list[str],
    signal_date: str,
    lookback: int = 60,
) -> list[dict]:
    """실거래 시작 전 과거 데이터로 포트폴리오 수익률 워밍업 히스토리를 생성한다.

    fills 기반 히스토리가 lookback에 미치지 못할 때 pykrx로 과거 주가를 조회해
    동일가중 포트폴리오 가치 시계열을 구성하여 반환한다.

    Parameters
    ----------
    tickers:
        현재 전략이 보유 중이거나 보유할 종목 코드 리스트.
    signal_date:
        기준 날짜 (형식: 'YYYY-MM-DD'). 이 날짜 이전 lookback 거래일을 조회한다.
    lookback:
        필요한 거래일 수. 기본 60일.

    Returns
    -------
    {'date': str, 'portfolio_value': float} 딕셔너리 리스트 (시간 오름차순).
    lookback + 1개의 포인트를 반환한다. 실패 시 빈 리스트.
    """
    try:
        from pykrx import stock as _pykrx_stock
    except ImportError:
        return []

    safe_lookback = max(5, int(lookback))
    # 비영업일 여유분 포함해 더 많은 거래일을 조회한다
    fetch_days = safe_lookback + 20

    try:
        end_dt = pd.Timestamp(signal_date)
        start_dt = end_dt - pd.tseries.offsets.BDay(fetch_days)
        start_date = start_dt.strftime("%Y%m%d")
        end_date = end_dt.strftime("%Y%m%d")
    except Exception:
        return []

    price_data: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            df = _pykrx_stock.get_market_ohlcv(start_date, end_date, ticker)
            if df is not None and not df.empty and "종가" in df.columns:
                price_data[ticker] = df["종가"].astype(float)
        except Exception:
            continue

    if not price_data:
        return []

    # 동일가중 포트폴리오 가치 시계열 구성
    # tail(safe_lookback + 1)로 결측행 전처리 후 정확한 개수를 확보한다
    df_prices = pd.DataFrame(price_data).dropna(how="all").ffill()
    if len(df_prices) < safe_lookback + 1:
        return []
    df_prices = df_prices.tail(safe_lookback + 1)

    df_returns = df_prices.pct_change().iloc[1:]
    portfolio_returns = df_returns.mean(axis=1)

    # 기준점 1.0에서 누적 포트폴리오 가치 산출 (lookback+1 포인트)
    cum_values: list[float] = [1.0]
    for r in portfolio_returns:
        cum_values.append(cum_values[-1] * (1.0 + float(r)))

    dates = list(df_prices.index)
    return [
        {"date": pd.Timestamp(dt).strftime("%Y-%m-%d"), "portfolio_value": float(v)}
        for dt, v in zip(dates, cum_values)
    ]
