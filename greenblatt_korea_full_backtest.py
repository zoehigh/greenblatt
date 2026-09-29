"""
그린블라트 응용 전략 - 한국 주식 백테스트 (실행 가능 버전)

필수 패키지 설치:
uv sync

실행 방법:
uv run greenblatt_korea_full_backtest.py

 데이터 출처:
 - FinanceDataReader: 주가 데이터
 - pykrx: 재무제표 및 시장 데이터

추가 기능:
 - `mixed_filter_profile='large_cap'` 옵션: 기존의 소형/중형 중심 스크리닝 대신 시가총액 상위 기업(상위 20% 기반) 및 하한선(`large_cap_min_mcap`)을 적용해 대형주 위주의 포트폴리오를 생성합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
import argparse
from defaults import DEFAULT_REBALANCE_MONTHS
from utils.env import env_get
from vol_targeting import compute_vol_target_ratio

try:
    from dotenv import find_dotenv, load_dotenv

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)
    else:
        project_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        load_dotenv(dotenv_path=project_env_path, override=False)
except ImportError:
    print("경고: python-dotenv가 설치되지 않았습니다.")
    print("설치 명령: uv sync")

try:
    import FinanceDataReader as fdr
    from pykrx import stock
    LIBRARIES_AVAILABLE = True
except ImportError:
    LIBRARIES_AVAILABLE = False
    print("경고: FinanceDataReader 또는 pykrx가 설치되지 않았습니다.")
    print("설치 명령: uv sync")

from stock_selector import KoreaStockSelector, ScreenStatus
from live_trading.strategy_config import BacktestConfig
from live_trading.execution import select_capital_constrained_stocks


# 일별 수익률 산정에서 제외할 date gap 임계값(일). 이보다 큰 간격의 수익률은
# 결측 구간/비영업일이 연속 거래처럼 변동성을 부풀리는 착시를 방지하기 위해 제외한다.
_GAP_EXCLUSION_DAYS = 4


class KoreaStockBacktest:
    """한국 주식 그린블라트 응용 전략 백테스트"""
    
    def __init__(self, start_date='2017-05-01', end_date='2025-04-30',
                 initial_capital=None, investment_ratio=None, num_stocks=None,
                 commission_fee_rate=None, tax_rate=None, rebalance_months=None,
                 rebalance_days=None,
                 strategy_mode=None, mixed_filter_profile=None,
                 sell_losers_enabled=None, kosdaq_target_ratio=None,
                 momentum_enabled=None, momentum_months=None, momentum_weight=None,
                 momentum_filter_enabled=None,
                 large_cap_min_mcap=None,
                 fundamental_source=None,
                 capital_constrained_selection_enabled=True,
                 capital_constrained_min_stocks=20,
                 capital_constrained_max_stocks=None,
                 cache_dir=None,
                 timing_enabled=True,
                 fundamental_cache_format=None,
                 fundamental_cache_max_entries=None,
                 slippage_bps: int = None,
                 vol_target_enabled: bool = None,
                 vol_target_sigma: float = None,
                 vol_target_lookback: int = None,
                 vol_target_min_ratio: float = None,
                 use_open_price: bool = None,
                 data_unavailable_policy: str = None):
        """
        Parameters:
        -----------
        start_date : str
            백테스트 시작일 (YYYY-MM-DD)
        end_date : str
            백테스트 종료일 (YYYY-MM-DD)
        initial_capital : int | None
            초기 투자금액 (원). None이면 환경 변수 또는 10,000,000원 사용
        investment_ratio : float | None
            투자 비율 (0.6 = 60%). None이면 환경 변수 또는 0.95 사용
        num_stocks : int | None
            보유 종목 수. None이면 환경 변수 또는 30 사용
        commission_fee_rate : float | None
            거래 수수료 비율. None이면 환경 변수 또는 0.0015 사용
        tax_rate : float | None
            양도소득세 비율. None이면 환경 변수 또는 0.002 사용
        rebalance_months : int | None
            리밸런싱 주기(개월). 12=연 1회, 6=반기, 3=분기
        strategy_mode : str | None
            종목 선정 모드 ('roe' 또는 'mixed'). None이면 환경 변수 또는 'mixed' 사용
        mixed_filter_profile : str | None
            mixed 모드 필터 프로파일. None이면 환경 변수 또는 'large_cap' 사용
        sell_losers_enabled : bool | None
            1년 보유 후 손실 종목 매도 사용 여부. None이면 SELL_LOSERS_ENABLED 환경변수 또는 True 사용
        kosdaq_target_ratio : float | None
            KOSDAQ 목표 비중(0~1). None이면 강제 비중을 사용하지 않음
        """
        self.start_date = start_date
        self.end_date = end_date
        
        # 1. 자산 및 비중 설정
        if initial_capital is None:
            self.initial_capital = int(env_get('INITIAL_CAPITAL', fallback_keys=['BACKTEST_INITIAL_CAPITAL'], default='10000000'))
        else:
            self.initial_capital = initial_capital
            
        if investment_ratio is None:
            self.investment_ratio = float(env_get('INVESTMENT_RATIO', fallback_keys=['BACKTEST_INVESTMENT_RATIO', 'LIVE_INVESTMENT_RATIO'], default='0.95'))
        else:
            self.investment_ratio = investment_ratio
            
        if num_stocks is None:
            self.num_stocks = int(env_get('NUM_STOCKS', fallback_keys=['BACKTEST_NUM_STOCKS', 'LIVE_NUM_STOCKS'], default='40'))
        else:
            self.num_stocks = num_stocks

        # 2. 비용 및 세금 설정
        if commission_fee_rate is None:
            self.commission_fee_rate = float(env_get('COMMISSION_FEE_RATE', fallback_keys=['BACKTEST_COMMISSION_FEE_RATE', 'LIVE_COMMISSION_FEE_RATE'], default='0.0015'))
        else:
            self.commission_fee_rate = commission_fee_rate
            
        if tax_rate is None:
            self.tax_rate = float(env_get('TAX_RATE', fallback_keys=['BACKTEST_TAX_RATE', 'LIVE_TAX_RATE'], default='0.002'))
        else:
            self.tax_rate = tax_rate

        # 3. 리밸런싱 주기 설정
        # rebalance_days: 우선순위 -> 인자 > REBALANCE_DAYS env > LIVE_REBALANCE_DAYS env
        self.rebalance_days = None
        if rebalance_days is None:
            env_days = env_get('REBALANCE_DAYS', fallback_keys=['REBALANCE_DAYS', 'LIVE_REBALANCE_DAYS'])
            if env_days not in (None, ""):
                try:
                    parsed_days = int(env_days)
                    if parsed_days > 0:
                        self.rebalance_days = parsed_days
                except Exception:
                    self.rebalance_days = None
        else:
            try:
                parsed_days = int(rebalance_days)
                if parsed_days > 0:
                    self.rebalance_days = parsed_days
            except Exception:
                self.rebalance_days = None

        # rebalance_months: 우선순위 -> 인자 > REBALANCE_MONTHS env > LIVE_REBALANCE_MONTHS env > 기본값(DEFAULT_REBALANCE_MONTHS)
        if rebalance_months is None:
            env_reb = env_get('REBALANCE_MONTHS', fallback_keys=['REBALANCE_MONTHS', 'LIVE_REBALANCE_MONTHS'])
            if env_reb not in (None, ""):
                try:
                    self.rebalance_months = int(env_reb)
                except Exception:
                    self.rebalance_months = DEFAULT_REBALANCE_MONTHS
            else:
                self.rebalance_months = DEFAULT_REBALANCE_MONTHS
        else:
            self.rebalance_months = int(rebalance_months)

        # 4. 전략 설정
        if strategy_mode is None:
            self.strategy_mode = env_get('STRATEGY_MODE', fallback_keys=['BACKTEST_STRATEGY_MODE', 'LIVE_STRATEGY_MODE'], default='mixed')
        else:
            self.strategy_mode = strategy_mode
            
        if mixed_filter_profile is None:
            self.mixed_filter_profile = env_get('MIXED_FILTER_PROFILE', fallback_keys=['BACKTEST_MIX_PROFILE', 'LIVE_MIXED_FILTER_PROFILE'], default='large_cap')
        else:
            self.mixed_filter_profile = mixed_filter_profile

        if sell_losers_enabled is None:
            env_sl = env_get('SELL_LOSERS_ENABLED', fallback_keys=['BACKTEST_SELL_LOSERS_ENABLED'], default='true')
            self.sell_losers_enabled = str(env_sl).lower() in {'true', '1', 'yes', 'y'}
        else:
            self.sell_losers_enabled = bool(sell_losers_enabled)
        # sell_losers hold 기간 설정 우선순위:
        # 1) SELL_LOSERS_HOLD_DAYS env (정수 일수)
        # 2) SELL_LOSERS_HOLD_REBALANCE_CYCLES env (정수, 리밸런스 주기 단위의 사이클 수)
        #    -> cycles * (rebalance_days if 설정되어 있으면 일수 기준, 아니면 rebalance_months*30로 근사)
        # 3) 기본값: 365
        self.sell_losers_hold_days = None
        env_hold_days = env_get('SELL_LOSERS_HOLD_DAYS')
        if env_hold_days not in (None, ''):
            try:
                parsed = int(env_hold_days)
                if parsed > 0:
                    self.sell_losers_hold_days = parsed
            except Exception:
                self.sell_losers_hold_days = None

        if self.sell_losers_hold_days is None:
            env_cycles = env_get('SELL_LOSERS_HOLD_REBALANCE_CYCLES')
            if env_cycles not in (None, ''):
                try:
                    cycles = int(env_cycles)
                    if cycles > 0:
                        if self.rebalance_days is not None and self.rebalance_days > 0:
                            base_days = int(self.rebalance_days)
                        else:
                            # 월 단위는 약 30일로 환산
                            base_days = int(self.rebalance_months * 30)
                        self.sell_losers_hold_days = max(1, cycles * base_days)
                except Exception:
                    self.sell_losers_hold_days = None

        if self.sell_losers_hold_days is None:
            self.sell_losers_hold_days = 365
        self.kosdaq_target_ratio = kosdaq_target_ratio
        
        # 5. 모멘텀 설정
        if momentum_enabled is None:
            env_mom_enabled = env_get('MOMENTUM_ENABLED', fallback_keys=['BACKTEST_MOMENTUM_ENABLED', 'LIVE_MOMENTUM_ENABLED'], default='true')
            self.momentum_enabled = str(env_mom_enabled).lower() in {'true', '1', 'yes', 'y'}
        else:
            self.momentum_enabled = momentum_enabled
            
        if momentum_months is None:
            self.momentum_months = int(env_get('MOMENTUM_MONTHS', fallback_keys=['BACKTEST_MOMENTUM_MONTHS', 'LIVE_MOMENTUM_MONTHS'], default='3'))
        else:
            self.momentum_months = momentum_months
            
        if momentum_weight is None:
            self.momentum_weight = float(env_get('MOMENTUM_WEIGHT', fallback_keys=['BACKTEST_MOMENTUM_WEIGHT', 'LIVE_MOMENTUM_WEIGHT'], default='0.60'))
        else:
            self.momentum_weight = momentum_weight
            
        if momentum_filter_enabled is None:
            env_mom_filter = env_get('MOMENTUM_FILTER_ENABLED', fallback_keys=['BACKTEST_MOMENTUM_FILTER_ENABLED', 'LIVE_MOMENTUM_FILTER_ENABLED'], default='true')
            self.momentum_filter_enabled = str(env_mom_filter).lower() in {'true', '1', 'yes', 'y'}
        else:
            self.momentum_filter_enabled = momentum_filter_enabled

        # Large Cap 및 데이터 소스
        if large_cap_min_mcap is None:
            env_lcap = env_get('LARGE_CAP_MIN_MCAP', fallback_keys=['BACKTEST_LARGE_CAP_MIN_MCAP', 'LIVE_LARGE_CAP_MIN_MCAP'])
            self.large_cap_min_mcap = float(env_lcap) if env_lcap else None
        else:
            self.large_cap_min_mcap = large_cap_min_mcap
            
        self.fundamental_source = str(fundamental_source or env_get('FUNDAMENTAL_SOURCE', fallback_keys=['BACKTEST_FUNDAMENTAL_SOURCE', 'LIVE_FUNDAMENTAL_SOURCE'], default='pykrx')).strip().lower()
        
        self.capital_constrained_selection_enabled = bool(capital_constrained_selection_enabled)
        self.capital_constrained_min_stocks = int(capital_constrained_min_stocks)
        self.capital_constrained_max_stocks = int(capital_constrained_max_stocks) if capital_constrained_max_stocks else int(self.num_stocks)
        
        # 6. 시스템 및 캐시 설정
        if cache_dir is None:
            self.cache_dir = env_get('CACHE_DIR', fallback_keys=['BACKTEST_CACHE_DIR'], default='results/cache')
        else:
            self.cache_dir = cache_dir
            
        self.timing_enabled = timing_enabled
        
        if fundamental_cache_format is None:
            self.fundamental_cache_format = env_get('FUNDAMENTAL_CACHE_FORMAT', fallback_keys=['BACKTEST_FUNDAMENTAL_CACHE_FORMAT'], default='parquet')
        else:
            self.fundamental_cache_format = fundamental_cache_format
            
        if fundamental_cache_max_entries is None:
            self.fundamental_cache_max_entries = int(env_get('FUNDAMENTAL_CACHE_MAX_ENTRIES', fallback_keys=['BACKTEST_FUNDAMENTAL_CACHE_MAX_ENTRIES'], default='16'))
        else:
            self.fundamental_cache_max_entries = max(1, int(fundamental_cache_max_entries))

        self.cache_version = {
            'fundamental_cache_v': 2,
            'momentum_cache_v': 1,
            'strategy_mode': self.strategy_mode,
            'momentum_months': self.momentum_months,
            'cache_format': self.fundamental_cache_format,
        }
        
        self.portfolio = {}
        self.cash = self.initial_capital
        self.portfolio_history = []
        self._data_missing_dates: list[str] = []
        self.trade_history = []

        # 7. 슬리피지 및 변동성 타겟팅
        if slippage_bps is None:
            # 실전 투자의 LIVE_ORDER_PRICE_OFFSET_BPS를 슬리피지로 대응
            self.slippage_bps = int(env_get('SLIPPAGE_BPS', fallback_keys=['BACKTEST_SLIPPAGE_BPS', 'LIVE_ORDER_PRICE_OFFSET_BPS'], default='30'))
        else:
            self.slippage_bps = int(slippage_bps)
        self.slippage_rate = float(self.slippage_bps) / 10000.0

        if vol_target_enabled is None:
            env_vol_enabled = env_get('VOL_TARGET_ENABLED', fallback_keys=['BACKTEST_VOL_TARGET_ENABLED', 'LIVE_VOL_TARGET_ENABLED'], default='true')
            self.vol_target_enabled = str(env_vol_enabled).lower() in {'true', '1', 'yes', 'y'}
        else:
            self.vol_target_enabled = bool(vol_target_enabled)
            
        if vol_target_sigma is None:
            self.vol_target_sigma = float(env_get('VOL_TARGET_SIGMA', fallback_keys=['BACKTEST_VOL_TARGET_SIGMA', 'LIVE_VOL_TARGET_SIGMA'], default='0.28'))
        else:
            self.vol_target_sigma = float(vol_target_sigma)
            
        if vol_target_lookback is None:
            self.vol_target_lookback = int(env_get('VOL_TARGET_LOOKBACK', fallback_keys=['BACKTEST_VOL_TARGET_LOOKBACK', 'LIVE_VOL_TARGET_LOOKBACK'], default='60'))
        else:
            self.vol_target_lookback = int(vol_target_lookback)
            
        if vol_target_min_ratio is None:
            self.vol_target_min_ratio = float(env_get('VOL_TARGET_MIN_RATIO', fallback_keys=['BACKTEST_VOL_TARGET_MIN_RATIO', 'LIVE_VOL_TARGET_MIN_RATIO'], default='0.65'))
        else:
            self.vol_target_min_ratio = float(vol_target_min_ratio)

        # 8. T 시가 체결 사용 여부 (false면 T-1 종가로 체결)
        if use_open_price is None:
            env_uop = env_get('USE_OPEN_PRICE', fallback_keys=['BACKTEST_USE_OPEN_PRICE'], default='false')
            self.use_open_price = str(env_uop).lower() in {'true', '1', 'yes', 'y'}
        else:
            self.use_open_price = bool(use_open_price)

        # 8-1. 데이터 조회 실패 정책: strict(기본) / hold_and_mark(진단용)
        if data_unavailable_policy is None:
            self.data_unavailable_policy = env_get(
                'DATA_UNAVAILABLE_POLICY',
                fallback_keys=['BACKTEST_DATA_UNAVAILABLE_POLICY'],
                default='strict',
            ).strip().lower()
        else:
            self.data_unavailable_policy = str(data_unavailable_policy).strip().lower()
        if self.data_unavailable_policy not in {'strict', 'hold_and_mark'}:
            raise ValueError(
                f"data_unavailable_policy는 'strict' 또는 'hold_and_mark'여야 합니다 "
                f"(받은 값: {self.data_unavailable_policy!r})"
            )

        self.industry_cache = {}
        self.momentum_cache = {}
        self.price_cache = {}
        self.fundamental_cache = OrderedDict()  # {date|market: DataFrame} LRU
        self._load_caches()
        self.selector = KoreaStockSelector(
            num_stocks=self.num_stocks,
            strategy_mode=self.strategy_mode,
            mixed_filter_profile=self.mixed_filter_profile,
            kosdaq_target_ratio=self.kosdaq_target_ratio,
            momentum_enabled=self.momentum_enabled,
            momentum_months=self.momentum_months,
            momentum_weight=self.momentum_weight,
            momentum_filter_enabled=self.momentum_filter_enabled,
            large_cap_min_mcap=self.large_cap_min_mcap,
            fundamental_source=self.fundamental_source,
            cache_dir=self.cache_dir,
            timing_enabled=self.timing_enabled,
            fundamental_cache_format=self.fundamental_cache_format,
            fundamental_cache_max_entries=self.fundamental_cache_max_entries,
        )

    def _cache_paths(self):
        return {
            'industry': os.path.join(self.cache_dir, 'industry_cache.json'),
            'momentum': os.path.join(self.cache_dir, 'momentum_cache.json'),
            'price': os.path.join(self.cache_dir, 'price_cache.json'),
            'fundamentals': os.path.join(self.cache_dir, 'fundamentals'),
            'meta': os.path.join(self.cache_dir, 'cache_meta.json')
        }

    @classmethod
    def from_config(cls, config: "BacktestConfig") -> "KoreaStockBacktest":
        """BacktestConfig로부터 KoreaStockBacktest 인스턴스를 생성한다.

        기존 개별 파라미터 방식의 __init__과 하위 호환성을 유지하면서,
        BacktestConfig 단일 객체로 인스턴스를 생성하는 팩토리 메서드.
        """
        bt = cls(
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital,
            investment_ratio=config.investment_ratio,
            num_stocks=config.num_stocks,
            commission_fee_rate=config.commission_fee_rate,
            tax_rate=config.tax_rate,
            rebalance_months=config.rebalance_months,
            rebalance_days=config.rebalance_days,
            strategy_mode=config.strategy_mode,
            mixed_filter_profile=config.mixed_filter_profile,
            sell_losers_enabled=config.sell_losers_enabled,
            kosdaq_target_ratio=config.kosdaq_target_ratio,
            momentum_enabled=config.momentum_enabled,
            momentum_months=config.momentum_months,
            momentum_weight=config.momentum_weight,
            momentum_filter_enabled=config.momentum_filter_enabled,
            large_cap_min_mcap=config.large_cap_min_mcap,
            fundamental_source=config.fundamental_source,
            capital_constrained_selection_enabled=config.capital_constrained_selection_enabled,
            capital_constrained_min_stocks=config.capital_constrained_min_stocks,
            capital_constrained_max_stocks=config.capital_constrained_max_stocks,
            slippage_bps=config.slippage_bps,
            vol_target_enabled=config.vol_target_enabled,
            vol_target_sigma=config.vol_target_sigma,
            vol_target_lookback=config.vol_target_lookback,
            vol_target_min_ratio=config.vol_target_min_ratio,
            use_open_price=config.use_open_price,
            data_unavailable_policy=config.data_unavailable_policy,
            cache_dir=config.cache_dir,
            timing_enabled=config.timing_enabled,
            fundamental_cache_format=config.fundamental_cache_format,
            fundamental_cache_max_entries=config.fundamental_cache_max_entries,
        )
        # __init__은 hold_days를 env에서 계산하지만, config 값이 있으면 우선한다.
        bt.sell_losers_hold_days = config.sell_losers_hold_days
        return bt

    def _validate_cache_version(self, loaded_meta):
        if not isinstance(loaded_meta, dict):
            return False
        for key, value in self.cache_version.items():
            if loaded_meta.get(key) != value:
                return False
        return True

    def _purge_fundamental_disk_cache(self):
        paths = self._cache_paths()
        fundamentals_dir = paths['fundamentals']
        if not os.path.isdir(fundamentals_dir):
            return
        for file_name in os.listdir(fundamentals_dir):
            if file_name.endswith('.parquet') or file_name.endswith('.csv'):
                try:
                    os.remove(os.path.join(fundamentals_dir, file_name))
                except Exception:
                    pass

    def _fundamental_cache_file(self, date_str, market):
        paths = self._cache_paths()
        fundamentals_dir = paths['fundamentals']
        preferred_ext = '.parquet' if self.fundamental_cache_format == 'parquet' else '.csv'
        preferred = os.path.join(fundamentals_dir, f'{date_str}_{market}{preferred_ext}')
        alternate_ext = '.csv' if preferred_ext == '.parquet' else '.parquet'
        alternate = os.path.join(fundamentals_dir, f'{date_str}_{market}{alternate_ext}')
        return preferred, alternate

    def _load_fundamental_frame(self, date_str, market):
        preferred, alternate = self._fundamental_cache_file(date_str, market)
        for path in [preferred, alternate]:
            if not os.path.exists(path):
                continue
            try:
                if path.endswith('.parquet'):
                    return pd.read_parquet(path)
                return pd.read_csv(path)
            except Exception:
                continue
        return None

    def _save_fundamental_frame(self, date_str, market, df):
        preferred, _ = self._fundamental_cache_file(date_str, market)
        try:
            if preferred.endswith('.parquet'):
                df.to_parquet(preferred, index=False)
            else:
                df.to_csv(preferred, index=False, encoding='utf-8-sig')
            return
        except Exception:
            pass

        fallback = preferred.replace('.parquet', '.csv')
        try:
            df.to_csv(fallback, index=False, encoding='utf-8-sig')
        except Exception:
            pass

    def _set_fundamental_cache_lru(self, cache_key, frame):
        if cache_key in self.fundamental_cache:
            self.fundamental_cache.move_to_end(cache_key)
            self.fundamental_cache[cache_key] = frame
            return
        self.fundamental_cache[cache_key] = frame
        if len(self.fundamental_cache) > self.fundamental_cache_max_entries:
            self.fundamental_cache.popitem(last=False)

    def _load_json_cache(self, path):
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass
        return {}

    def _save_json_cache(self, path, data):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            print(f"캐시 저장 실패 ({path}): {e}")

    def _load_caches(self):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            paths = self._cache_paths()
            os.makedirs(paths['fundamentals'], exist_ok=True)

            loaded_meta = self._load_json_cache(paths['meta'])
            if not self._validate_cache_version(loaded_meta):
                print("[CACHE] cache version mismatch detected, invalidating incompatible caches")
                self.momentum_cache = {}
                self.fundamental_cache = OrderedDict()
                self._purge_fundamental_disk_cache()

            self.industry_cache = self._load_json_cache(paths['industry'])
            if len(self.momentum_cache) == 0:
                self.momentum_cache = self._load_json_cache(paths['momentum'])
            self.price_cache = self._load_json_cache(paths['price'])
            print(
                f"[CACHE] loaded: industry={len(self.industry_cache):,}, "
                f"momentum={len(self.momentum_cache):,}, price={len(self.price_cache):,}"
            )
        except Exception as e:
            print(f"[CACHE] load failed: {e}")
            self.industry_cache = {}
            self.momentum_cache = {}
            self.price_cache = {}
            self.fundamental_cache = {}

    def _persist_caches(self):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            paths = self._cache_paths()
            os.makedirs(paths['fundamentals'], exist_ok=True)
            self._save_json_cache(paths['industry'], self.industry_cache)
            self._save_json_cache(paths['momentum'], self.momentum_cache)
            self._save_json_cache(paths['price'], self.price_cache)
            self._save_json_cache(paths['meta'], self.cache_version)

            # Fundamental 캐시를 Parquet/CSV로 저장
            for cache_key, df in self.fundamental_cache.items():
                try:
                    parts = cache_key.split('|')
                    if len(parts) == 2:
                        date_str, market = parts
                        self._save_fundamental_frame(date_str, market, df)
                except Exception:
                    pass
            print(
                f"[CACHE] saved: industry={len(self.industry_cache):,}, "
                f"momentum={len(self.momentum_cache):,}, "
                f"fundamental={len(self.fundamental_cache):,}"
            )
        except Exception as e:
            print(f"[CACHE] save failed: {e}")

    def _log_timing(self, label, elapsed_sec, extra=''):
        if not self.timing_enabled:
            return
        if extra:
            print(f"  [TIME] {label}: {elapsed_sec:.3f}s ({extra})")
        else:
            print(f"  [TIME] {label}: {elapsed_sec:.3f}s")
        
    def get_market_tickers(self, date=None):
        """KOSPI + KOSDAQ 상장 종목 리스트 가져오기"""
        if hasattr(self, 'selector') and self.selector is not None:
            try:
                return self.selector.get_market_tickers(date)
            except Exception:
                pass

        if not LIBRARIES_AVAILABLE:
            return []
        
        try:
            if date is None:
                date = datetime.now().strftime('%Y%m%d')
            else:
                date = date.replace('-', '')
            
            tickers_kospi = stock.get_market_ticker_list(date=date, market="KOSPI")
            tickers_kosdaq = stock.get_market_ticker_list(date=date, market="KOSDAQ")
            return list(set(tickers_kospi + tickers_kosdaq))  # 중복 제거
        except Exception as e:
            print(f"종목 리스트 조회 실패: {e}")
            return []
    
    def _get_nearest_trading_date(self, date_str):
        """공휴일/주말이면 이후 가장 가까운 영업일 반환 (yyyymmdd 형식).
        get_nearest_business_day_in_a_week는 7일 제한이 있어
        추석 등 긴 연휴(10일+)에서 오동작한다.
        실제 거래 데이터(get_market_ohlcv_by_date)로 30일 범위를 탐색한다.
        """
        from datetime import datetime, timedelta
        dt = datetime.strptime(date_str, '%Y%m%d')
        end_str = (dt + timedelta(days=30)).strftime('%Y%m%d')
        try:
            df = stock.get_market_ohlcv_by_date(date_str, end_str, '005930')
            if not df.empty:
                return df.index[0].strftime('%Y%m%d')
        except Exception:
            pass
        # fallback: 최대 30일 탐색 (7일은 긴 연휴 대응 불가)
        for i in range(30):
            candidate = (dt + timedelta(days=i)).strftime('%Y%m%d')
            for market in ['KOSPI', 'KOSDAQ']:
                try:
                    df_test = stock.get_market_fundamental_by_ticker(candidate, market=market)
                    if not df_test.empty and df_test.iloc[:, 0].sum() != 0:
                        return candidate
                except Exception:
                    pass
        return date_str

    def _get_previous_trading_date(self, date_str):
        """반드시 T-1 이전의 가장 가까운 영업일 반환 (yyyymmdd 형식).
        get_nearest_business_day_in_a_week는 7일 제한이 있어
        추석 등 긴 연휴(10일+)에서 오동작한다.
        실제 거래 데이터(get_market_ohlcv_by_date)로 30일 범위를 탐색한다.
        """
        dt = datetime.strptime(date_str, '%Y%m%d')
        prev_dt = dt - timedelta(days=1)
        end_str = prev_dt.strftime('%Y%m%d')
        start_str = (prev_dt - timedelta(days=30)).strftime('%Y%m%d')
        try:
            df = stock.get_market_ohlcv_by_date(start_str, end_str, '005930')
            if not df.empty:
                return df.index[-1].strftime('%Y%m%d')
        except Exception:
            pass
        # fallback: 최대 30일 탐색 (7일은 긴 연휴 대응 불가)
        for i in range(30):
            candidate = (prev_dt - timedelta(days=i)).strftime('%Y%m%d')
            for market in ['KOSPI', 'KOSDAQ']:
                try:
                    df_test = stock.get_market_fundamental_by_ticker(candidate, market=market)
                    if not df_test.empty and df_test.iloc[:, 0].sum() != 0:
                        return candidate
                except Exception:
                    pass
        return end_str

    def rebalance(self, selected_stocks, rebalance_date, investment_ratio=None):
        """포트폴리오 리밸런싱"""
        commission_rate = self.commission_fee_rate
        tax_rate = self.tax_rate
        ratio_to_use = self.investment_ratio if investment_ratio is None else float(investment_ratio)
        # 부분 리밸런싱: 기존 보유 중 선정된 종목은 유지/조정, 제외 종목만 매도
        if len(selected_stocks) == 0:
            return

        # 가격 맵 (선정된 종목의 기준가격)
        price_map = selected_stocks.set_index('ticker')['close'].to_dict()

        # 1) 매도: 포트폴리오에 있으나 이번에 선정되지 않은 종목은 전량 매도
        sell_total = 0
        to_remove = []
        for ticker, position in list(self.portfolio.items()):
            if ticker not in price_map:
                sell_price = position.get('current_price', position['buy_price'])
                # 슬리피지 반영 실행가 (판매자는 실수령액이 소폭 감소함)
                exec_sell_price = sell_price * (1 - self.slippage_rate)
                gross_sell_amount = position['shares'] * exec_sell_price
                commission = gross_sell_amount * commission_rate
                tax = gross_sell_amount * tax_rate
                total_costs = commission + tax
                net_sell_amount = gross_sell_amount - total_costs
                self.cash += net_sell_amount
                sell_total += net_sell_amount

                self.trade_history.append({
                    'date': rebalance_date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': position['shares'],
                    'price': sell_price,
                    'exec_price': exec_sell_price,
                    'amount': net_sell_amount,
                    'gross_amount': gross_sell_amount,
                    'commission': commission,
                    'tax': tax,
                    'fee': total_costs
                })

                to_remove.append(ticker)

        for t in to_remove:
            del self.portfolio[t]

        # 2) 목표 자산 할당 기반 계산 (현재 포트폴리오 가치 기준)
        total_value = self.get_portfolio_value()
        invest_amount = total_value * ratio_to_use
        per_stock_amount = invest_amount / len(selected_stocks) if len(selected_stocks) > 0 else 0

        # 2-1) 목표 수량 사전 계산 및 잔액 순환 재배분
        _denom = (1 + commission_rate) * (1 + self.slippage_rate)
        _target_shares_map: dict[str, int] = {}
        for _, _s in selected_stocks.iterrows():
            try:
                _p = float(_s['close'])
                if pd.isna(_p) or _p <= 0:
                    continue
            except Exception:
                continue
            _target_shares_map[_s['ticker']] = int(per_stock_amount / (_p * _denom))
        _remaining = invest_amount - sum(
            _target_shares_map.get(t, 0) * float(price_map[t]) * _denom
            for t in _target_shares_map
        )
        _changed = True
        while _changed:
            _changed = False
            for _, _s in selected_stocks.iterrows():
                _t = _s['ticker']
                if _t not in _target_shares_map:
                    continue
                _cost = float(_s['close']) * _denom
                if _remaining >= _cost:
                    _target_shares_map[_t] += 1
                    _remaining -= _cost
                    _changed = True

        # 3) 목표 수량 산정 및 매매 (보유 종목은 조정, 신규는 매수)
        for _, stock in selected_stocks.iterrows():
            ticker = stock['ticker']
            price = stock['close']
            # 가격 정보가 없거나 비정상이면 해당 종목 건너뜀
            try:
                if pd.isna(price) or float(price) <= 0:
                    continue
            except Exception:
                continue
            # 목표 수량 (수수료+슬리피지 고려, 잔액 재배분 적용)
            target_shares = _target_shares_map.get(ticker, 0)

            if ticker in self.portfolio:
                position = self.portfolio[ticker]
                current_shares = position['shares']
                # 조정: 과다 보유면 일부 매도, 부족하면 추가 매수
                if target_shares < current_shares:
                    sell_shares = current_shares - target_shares
                    sell_price = position.get('current_price', position['buy_price'])
                    # 슬리피지 반영 실행가 (판매자는 실수령액이 소폭 감소함)
                    exec_sell_price = sell_price * (1 - self.slippage_rate)
                    gross_sell_amount = sell_shares * exec_sell_price
                    commission = gross_sell_amount * commission_rate
                    tax = gross_sell_amount * tax_rate
                    total_costs = commission + tax
                    net_sell_amount = gross_sell_amount - total_costs
                    self.cash += net_sell_amount

                    position['shares'] = current_shares - sell_shares
                    self.trade_history.append({
                        'date': rebalance_date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'shares': sell_shares,
                        'price': sell_price,
                        'exec_price': exec_sell_price,
                        'amount': net_sell_amount,
                        'gross_amount': gross_sell_amount,
                        'commission': commission,
                        'tax': tax,
                        'fee': total_costs
                    })
                    # 포지션이 0이 되면 제거
                    if position['shares'] <= 0:
                        del self.portfolio[ticker]
                elif target_shares > current_shares:
                    buy_shares = target_shares - current_shares
                    # 슬리피지 반영 실행가 (구매자는 실제로 소폭 더 지불함)
                    exec_buy_price = price * (1 + self.slippage_rate)
                    gross_buy_amount = buy_shares * exec_buy_price
                    buy_commission = gross_buy_amount * commission_rate
                    total_buy_cost = gross_buy_amount + buy_commission
                    # 현금 부족 시 구매수량 조정 (실행가격+수수료 반영)
                    if total_buy_cost > self.cash:
                        denom = price * (1 + self.slippage_rate) * (1 + commission_rate)
                        affordable_shares = int(self.cash / denom)
                        buy_shares = max(0, affordable_shares)
                        exec_buy_price = price * (1 + self.slippage_rate)
                        gross_buy_amount = buy_shares * exec_buy_price
                        buy_commission = gross_buy_amount * commission_rate
                        total_buy_cost = gross_buy_amount + buy_commission

                    if buy_shares > 0:
                        self.cash -= total_buy_cost
                        # 가중 평균 매입 단가 적용
                        prev_shares = position['shares']
                        prev_price = position.get('buy_price', price)
                        new_total_shares = prev_shares + buy_shares
                        new_buy_price = ((prev_shares * prev_price) + (buy_shares * price)) / new_total_shares
                        position['shares'] = new_total_shares
                        position['buy_price'] = new_buy_price
                        position['current_price'] = price

                        self.trade_history.append({
                            'date': rebalance_date,
                            'ticker': ticker,
                            'action': 'BUY',
                            'shares': buy_shares,
                            'price': price,
                            'exec_price': exec_buy_price,
                            'amount': total_buy_cost,
                            'gross_amount': gross_buy_amount,
                            'commission': buy_commission,
                            'tax': 0,
                            'fee': buy_commission
                        })
                else:
                    # 목표수량과 동일하면 가격만 업데이트
                    position['current_price'] = price
            else:
                # 신규 매수
                buy_shares = target_shares
                if buy_shares <= 0:
                    continue
                exec_buy_price = price * (1 + self.slippage_rate)
                gross_buy_amount = buy_shares * exec_buy_price
                buy_commission = gross_buy_amount * commission_rate
                total_buy_cost = gross_buy_amount + buy_commission
                if total_buy_cost > self.cash:
                    denom = price * (1 + self.slippage_rate) * (1 + commission_rate)
                    affordable_shares = int(self.cash / denom)
                    buy_shares = max(0, affordable_shares)
                    exec_buy_price = price * (1 + self.slippage_rate)
                    gross_buy_amount = buy_shares * exec_buy_price
                    buy_commission = gross_buy_amount * commission_rate
                    total_buy_cost = gross_buy_amount + buy_commission

                if buy_shares > 0:
                    self.cash -= total_buy_cost
                    self.portfolio[ticker] = {
                        'ticker': ticker,
                        'shares': buy_shares,
                        'buy_price': price,
                        'buy_date': rebalance_date,
                        'current_price': price
                    }

                    self.trade_history.append({
                        'date': rebalance_date,
                        'ticker': ticker,
                        'action': 'BUY',
                        'shares': buy_shares,
                        'price': price,
                        'exec_price': exec_buy_price,
                        'amount': total_buy_cost,
                        'gross_amount': gross_buy_amount,
                        'commission': buy_commission,
                        'tax': 0,
                        'fee': buy_commission
                    })
    
    def update_portfolio_prices(self, date):
        """포트폴리오 보유 종목 가격 업데이트"""
        if not LIBRARIES_AVAILABLE:
            return
        
        date_str = date.replace('-', '')
        cache_hit = 0
        cache_miss = 0
        t_start = time.perf_counter()
        
        for ticker in list(self.portfolio.keys()):
            cache_key = f"{date_str}|{ticker}"
            if cache_key in self.price_cache:
                self.portfolio[ticker]['current_price'] = self.price_cache[cache_key]
                cache_hit += 1
                continue
            try:
                df = stock.get_market_ohlcv(date_str, date_str, ticker)
                if not df.empty:
                    close_price = float(df['종가'].iloc[0])
                    self.portfolio[ticker]['current_price'] = close_price
                    self.price_cache[cache_key] = close_price
                    cache_miss += 1
            except:
                pass

        self._log_timing(
            'portfolio.price_update',
            time.perf_counter() - t_start,
            extra=f"hit={cache_hit}, miss={cache_miss}, holdings={len(self.portfolio)}"
        )
    
    def sell_losers(self, current_date):
        """1년 보유 후 손실 종목 매도"""
        commission_rate = self.commission_fee_rate
        tax_rate = self.tax_rate
        current_date_obj = datetime.strptime(current_date, '%Y-%m-%d')
        tickers_to_sell = []
        
        for ticker, position in self.portfolio.items():
            buy_date_obj = datetime.strptime(position['buy_date'], '%Y-%m-%d')
            holding_days = (current_date_obj - buy_date_obj).days
            
            # 설정된 보유기간 이상 보유
            if holding_days >= int(self.sell_losers_hold_days):
                current_price = position['current_price']
                buy_price = position['buy_price']
                return_rate = (current_price - buy_price) / buy_price
                
                # 손실이면 매도
                if return_rate < 0:
                    tickers_to_sell.append(ticker)
        
        # 매도 실행
        for ticker in tickers_to_sell:
            position = self.portfolio[ticker]
            # 슬리피지 반영 실행가 (판매자는 실수령액이 소폭 감소함)
            exec_sell_price = position['current_price'] * (1 - self.slippage_rate)
            gross_sell_amount = position['shares'] * exec_sell_price
            commission = gross_sell_amount * commission_rate
            tax = gross_sell_amount * tax_rate
            total_costs = commission + tax
            net_sell_amount = gross_sell_amount - total_costs
            self.cash += net_sell_amount
            
            # 매도 기록
            self.trade_history.append({
                'date': current_date,
                'ticker': ticker,
                'action': 'SELL_LOSS',
                'shares': position['shares'],
                'price': position['current_price'],
                'exec_price': exec_sell_price,
                'amount': net_sell_amount,
                'gross_amount': gross_sell_amount,
                'commission': commission,
                'tax': tax,
                'fee': total_costs,
                'return': (position['current_price'] - position['buy_price']) / position['buy_price']
            })
            
            del self.portfolio[ticker]
    
    def get_portfolio_value(self):
        """현재 포트폴리오 총 가치"""
        stock_value = sum(
            pos['shares'] * pos['current_price'] 
            for pos in self.portfolio.values()
        )
        return self.cash + stock_value
    
    def _fetch_period_close_prices(self, tickers: list, from_date: str, to_date: str) -> 'pd.DataFrame':
        """리밸런싱 구간 내 보유 종목 일별 종가 DataFrame 반환 (index=날짜, columns=티커)"""
        if not tickers:
            return pd.DataFrame()
        from_str = from_date.replace('-', '')
        to_str = to_date.replace('-', '')
        price_data: dict = {}
        for ticker in tickers:
            try:
                df = stock.get_market_ohlcv(from_str, to_str, ticker)
                if df is not None and not df.empty and '종가' in df.columns:
                    series = df['종가'].astype(float)
                    price_data[ticker] = series
                    # price_cache 업데이트 (재실행 시 재조회 방지)
                    for dt_idx, val in series.items():
                        ck = f"{dt_idx.strftime('%Y%m%d')}|{ticker}"
                        if ck not in self.price_cache:
                            self.price_cache[ck] = float(val)
            except Exception:
                pass
        if not price_data:
            return pd.DataFrame()
        result = pd.DataFrame(price_data)
        result = result.ffill()  # 거래 정지/결측치는 직전 종가로 채움
        return result

    def run_backtest(self):
        """백테스트 실행"""
        
        if not LIBRARIES_AVAILABLE:
            print("필요한 라이브러리가 설치되지 않았습니다.")
            return None
        
        print("\n" + "="*80)
        print("그린블라트 응용 전략 백테스트 시작")
        print("="*80)
        print(f"기간: {self.start_date} ~ {self.end_date}")
        print(f"초기자본: {self.initial_capital:,}원")
        print(f"투자비율: {self.investment_ratio*100}%")
        print(f"거래수수료: {self.commission_fee_rate*100:.2f}%")
        print(f"세금: {self.tax_rate*100:.2f}%")
        print(f"슬리피지: {self.slippage_bps} bps")
        print(f"보유종목수: {self.num_stocks}개")
        if self.rebalance_days is not None and self.rebalance_days > 0:
            print(f"리밸런싱주기: {self.rebalance_days}일")
        else:
            print(f"리밸런싱주기: {self.rebalance_months}개월")
        print(f"선정모드: {self.strategy_mode}")
        print(f"펀더멘털 소스: {self.fundamental_source}")
        if self.strategy_mode == 'mixed':
            print(f"필터프로파일: {self.mixed_filter_profile}")
        if self.kosdaq_target_ratio is not None:
            print(f"KOSDAQ 목표 비중: {self.kosdaq_target_ratio*100:.0f}%")
        print(f"손실매도: {'ON' if self.sell_losers_enabled else 'OFF'}")
        if self.sell_losers_enabled:
            print(f"손실매도 기준: {self.sell_losers_hold_days}일 보유 후 손실 시 매도")
        # 모멘텀 및 관련 옵션 상태 출력
        print(
            f"모멘텀: {'ON' if self.momentum_enabled else 'OFF'} | "
            f"기간={self.momentum_months}개월 | "
            f"가중치={self.momentum_weight} | "
            f"모멘텀필터: {'ON' if self.momentum_filter_enabled else 'OFF'}"
        )
        # 변동성 타게팅 및 자본제약 관련 설정
        print(f"변동성타게팅: {'ON' if self.vol_target_enabled else 'OFF'} (σ_target={self.vol_target_sigma}, lookback={self.vol_target_lookback}, min_ratio={self.vol_target_min_ratio})")
        print(f"자본제약선택 적용: {'ON' if self.capital_constrained_selection_enabled else 'OFF'} (min_stocks={self.capital_constrained_min_stocks}, max_stocks={self.capital_constrained_max_stocks})")
        print(f"체결기준: {'T 시가 (USE_OPEN_PRICE=true)' if self.use_open_price else 'T 종가 (장마감 동시호가)'}")
        print("="*80)
        total_start = time.perf_counter()
        
        # 리밸런싱 날짜 생성 (일 단위가 설정되면 일 단위 우선)
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        rebalance_dates = []
        current_date = pd.Timestamp(start)
        while current_date <= pd.Timestamp(end):
            rebalance_dates.append(current_date.strftime('%Y-%m-%d'))
            if self.rebalance_days is not None and self.rebalance_days > 0:
                current_date = current_date + pd.DateOffset(days=self.rebalance_days)
            else:
                current_date = current_date + pd.DateOffset(months=self.rebalance_months)
        
        # 백테스트 실행
        # 기록일 중복 방지용 전역 집합 (리밸런싱/일별 루프/최종평가 공용, boundary double-record 방지)
        recorded_dates: set = set()
        # 인스턴스 재사용 시 이전 실행의 결측 기록이 fail-fast에 오염되지 않도록 초기화
        self._data_missing_dates.clear()
        for i, rebal_date in enumerate(rebalance_dates):
            t_rebal_start = time.perf_counter()
            scheduled_date = rebal_date

            # T: 실제 주문 체결일 (장 시작 동시호가)
            execution_date = self.selector.nearest_trading_date(scheduled_date)
            execution_date_fmt = datetime.strptime(execution_date, '%Y%m%d').strftime('%Y-%m-%d')
            # T-1: 전일 종가 기준 종목 선정일 (장 마감 후 스크리닝)
            selection_date = self.selector.previous_trading_date(execution_date_fmt)
            selection_date_fmt = datetime.strptime(selection_date, '%Y%m%d').strftime('%Y-%m-%d')

            print(f"\n[{i+1}/{len(rebalance_dates)}] {scheduled_date} 리밸런싱 "
                  f"(선정: {selection_date_fmt} 종가, 체결: {execution_date_fmt} {'시가' if self.use_open_price else '종가'})")

            # 종목 선정: T-1 종가 기준 펀더멘탈로 스크리닝
            t_select_start = time.perf_counter()
            selected_stocks = self.selector.select_stocks(selection_date_fmt)
            self._log_timing('rebalance.select_stocks', time.perf_counter() - t_select_start)
            effective_date = execution_date_fmt

            # 스크리닝 상태 확인 (ScreenStatus): 데이터 부재(DATA_UNAVAILABLE)와 선정 불가
            # (NO_ELIGIBLE_STOCKS)를 구분한다. 구형 selector/테스트 더미에는 속성이
            # 없으므로 OK로 간주한다.
            screen_status = getattr(self.selector, 'last_screen_status', ScreenStatus.OK)
            # OK로 간주됐으나 df가 비어 있으면(테스트 더미 등) NO_ELIGIBLE로 흡수 —
            # OK 경로는 'ticker'/'close' 컬럼을 가정하므로 KeyError 방지.
            if screen_status == ScreenStatus.OK and selected_stocks.empty:
                screen_status = ScreenStatus.NO_ELIGIBLE_STOCKS

            # OK 경로에서 자본제약 선택으로 매수 가능 종목이 없어지면 NO_ELIGIBLE로 하위 흡수
            if screen_status == ScreenStatus.OK and self.capital_constrained_selection_enabled:
                holdings_map = {ticker: int(pos.get('shares', 0)) for ticker, pos in self.portfolio.items()}
                selected_stocks, alloc_meta = select_capital_constrained_stocks(
                    selected=selected_stocks,
                    holdings=holdings_map,
                    cash=self.cash,
                    investment_ratio=self.investment_ratio,
                    commission_fee_rate=self.commission_fee_rate,
                    max_stocks=self.capital_constrained_max_stocks,
                    min_stocks=self.capital_constrained_min_stocks,
                    slippage_rate=self.slippage_rate,
                )
                print(
                    "  [ALLOC] 자본제약 적용: "
                    f"before={int(alloc_meta['selected_before'])}, "
                    f"after={int(alloc_meta['selected_after'])}, "
                    f"k={int(alloc_meta['k_chosen'])}, "
                    f"주식당={float(alloc_meta['per_stock_amount']):,.0f}원"
                )
                if selected_stocks.empty:
                    print("  자본 제약으로 매수 가능한 종목이 없습니다.")
                    # NOTE: 자본제약으로 매수 불가 → 아래 NO_ELIGIBLE 경로(보유 유지)로 처리
                    screen_status = ScreenStatus.NO_ELIGIBLE_STOCKS

            # --- OK 경로 (정상 리밸런싱): 체결가 결정 + 손실매도 + 변동성타게팅 + 재조정 ---
            if screen_status == ScreenStatus.OK:
                print(f"  선정 종목: {len(selected_stocks)}개")

                # 체결가 결정: T-1 종가 기준 fallback 맵 구성
                selected_tickers = selected_stocks['ticker'].tolist()
                portfolio_tickers = list(self.portfolio.keys()) if i > 0 else []
                fallback_close: dict = selected_stocks.set_index('ticker')['close'].to_dict()
                for pt in portfolio_tickers:
                    if pt not in fallback_close:
                        fallback_close[pt] = self.portfolio[pt].get(
                            'current_price', self.portfolio[pt].get('buy_price', 0.0)
                        )
    
                if self.use_open_price:
                    # T 시가 일괄 조회: selected 종목 + 기존 보유 종목
                    all_tickers_for_open = list(set(selected_tickers + portfolio_tickers))
                    t_open_start = time.perf_counter()
                    open_prices, open_fallback = self.selector.get_open_prices(
                        all_tickers_for_open,
                        execution_date_fmt,
                        fallback_prices=fallback_close,
                    )
                    self._log_timing('rebalance.open_prices', time.perf_counter() - t_open_start,
                                     extra=f"tickers={len(all_tickers_for_open)}")
                    # fallback 종목 선정/보유 구분 로그
                    if open_fallback:
                        selected_set = set(selected_tickers)
                        fb_selected = [t for t in open_fallback if t in selected_set]
                        fb_hold = [t for t in open_fallback if t not in selected_set]
                        if fb_selected:
                            print(f"  [OPEN] ⚠ fallback(선정종목): {len(fb_selected)} tickers={fb_selected}")
                        if fb_hold:
                            print(f"  [OPEN] fallback(보유종목): {len(fb_hold)} tickers={fb_hold}")
                    # selected_stocks['close']를 T 시가로 교체
                    selected_stocks = selected_stocks.copy()
                    selected_stocks['close'] = selected_stocks['ticker'].map(
                        lambda t: open_prices.get(t, fallback_close.get(t))
                    )
                    # 보유 종목 current_price를 T 시가로 갱신
                    if i > 0 and len(self.portfolio) > 0:
                        for ticker, position in self.portfolio.items():
                            op = open_prices.get(ticker)
                            if op and op > 0:
                                position['current_price'] = op
                else:
                    # USE_OPEN_PRICE=false: T 종가로 체결 (장마감 동시호가 매칭)
                    # 실전 15:20 동시호가 주문과 동일한 기준 — 백테스트/실전 일관성 확보
                    all_tickers_for_close = list(set(selected_tickers + portfolio_tickers))
                    t_close_start = time.perf_counter()
                    close_prices, close_fallback = self.selector.get_close_prices(
                        all_tickers_for_close,
                        execution_date_fmt,
                        fallback_prices=fallback_close,
                    )
                    self._log_timing('rebalance.close_prices', time.perf_counter() - t_close_start,
                                     extra=f"tickers={len(all_tickers_for_close)}")
                    # fallback 종목 선정/보유 구분 로그
                    if close_fallback:
                        selected_set = set(selected_tickers)
                        fb_selected = [t for t in close_fallback if t in selected_set]
                        fb_hold = [t for t in close_fallback if t not in selected_set]
                        if fb_selected:
                            print(f"  [CLOSE] ⚠ fallback(선정종목): {len(fb_selected)} tickers={fb_selected}")
                        if fb_hold:
                            print(f"  [CLOSE] fallback(보유종목): {len(fb_hold)} tickers={fb_hold}")
                    # selected_stocks['close']를 T 종가로 교체
                    selected_stocks = selected_stocks.copy()
                    selected_stocks['close'] = selected_stocks['ticker'].map(
                        lambda t: close_prices.get(t, fallback_close.get(t))
                    )
                    # 보유 종목 current_price를 T 종가로 갱신
                    if i > 0 and len(self.portfolio) > 0:
                        for ticker, position in self.portfolio.items():
                            cp = close_prices.get(ticker)
                            if cp and cp > 0:
                                position['current_price'] = cp
                    print(f"  [CLOSE] T 종가로 체결 (tickers={len(selected_tickers)})")
    
                # 손실 종목 매도 (첫 리밸런싱 제외)
                if i > 0 and self.sell_losers_enabled:
                    t_sell_start = time.perf_counter()
                    self.sell_losers(effective_date)
                    self._log_timing('rebalance.sell_losers', time.perf_counter() - t_sell_start)
                
                # 변동성 타게팅: 실현 변동성 기반으로 유효 투자비율 동적 조정
                vol_decision = compute_vol_target_ratio(
                    portfolio_history=self.portfolio_history,
                    base_ratio=self.investment_ratio,
                    enabled=self.vol_target_enabled,
                    sigma_target=self.vol_target_sigma,
                    lookback_days=self.vol_target_lookback,
                    min_ratio=self.vol_target_min_ratio,
                )
                if self.vol_target_enabled:
                    print(
                        f"  [VOL-TARGET] reason={vol_decision.reason}, "
                        f"σ_realized={f'{vol_decision.sigma_realized*100:.1f}%' if vol_decision.sigma_realized is not None else 'N/A'}, "
                        f"σ_target={vol_decision.sigma_target*100:.0f}%, "
                        f"multiplier={vol_decision.multiplier:.3f}, "
                        f"base={vol_decision.base_ratio:.4f} → effective={vol_decision.effective_ratio:.4f}"
                    )
                effective_investment_ratio = vol_decision.effective_ratio
    
                # 리밸런싱
                t_exec_start = time.perf_counter()
                self.rebalance(selected_stocks, effective_date, investment_ratio=effective_investment_ratio)
                self._log_timing('rebalance.execute', time.perf_counter() - t_exec_start)
            
            # 포트폴리오 가치 기록 (상태별; OK/NO_ELIGIBLE/DATA 모두 동일 add+append 패턴)
            # T 시가/종가 체결 시 실제 체결일(execution_date)을 기준으로 기록해야
            # daily 모니터링에서 "매수 전 가격"이 섞이는 zig-zag 오류를 방지한다.
            # effective_date(T-1)로 기록하면 T-1→T-1+1→T 순서로 가짜 수익률이 생겨
            # σ_realized 가 60~100%로 부풀려 볼타게팅이 항상 65% 투자로 제한된다.
            record_date = execution_date_fmt

            if screen_status == ScreenStatus.DATA_UNAVAILABLE:
                self._data_missing_dates.append(record_date)
                if self.data_unavailable_policy == 'strict':
                    print("=" * 80)
                    print(f"⚠ 데이터 조회 실패: {record_date} 리밸런싱 건너뜀(보유 유지)")
                    print("=" * 80)
                    # strict(기본) 정책: 첫 DATA_UNAVAILABLE 발생 즉시 중단.
                    # 보유 유지로 이어지는 결측 구간이 성과 지표에 섞이지 않도록 한다.
                    raise RuntimeError(
                        f"DATA_UNAVAILABLE (strict): {record_date} — 리밸런싱 데이터 조회 실패 "
                        f"(selector status={screen_status.value!r}). "
                        "거짓 성과 지표를 생성하지 않습니다. "
                        "data_unavailable_policy='hold_and_mark'로 설정하면 보유 유지 마크로 계속할 수 있습니다."
                    )
                print("=" * 80)
                print(f"⚠ 데이터 조회 실패: {record_date} 리밸런싱 건너뜀(보유 유지) [hold_and_mark]")
                print("=" * 80)
            elif screen_status == ScreenStatus.NO_ELIGIBLE_STOCKS:
                # NOTE: 선정 불가(또는 자본제약으로 매수 불가) 시 리밸런싱뿐 아니라 손실매도
                #       (sell_losers)/변동성타게팅까지 함께 스킵한다. 데이터 없이 매도 판단을
                #       내리는 대신 보유 유지가 보수적으로 안전하다는 트레이드오프를 따른다.
                print("  선정 종목이 없습니다. 보유 유지 (리밸런싱/손실매도/변동성타게팅 건너뜀)")

            # 일별 루프용 가격 조회 (hold_and_mark는 DATA_UNAVAILABLE 분기에서 단일 fetch 재사용;
            # _fetch_period_close_prices는 price_cache만 쓰고 읽지 않으므로 중복 API 호출이 없다)
            next_boundary = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else self.end_date
            tickers_held = list(self.portfolio.keys())
            shares_map = {t: self.portfolio[t]['shares'] for t in tickers_held}
            fallback_prices = {t: self.portfolio[t].get('current_price', 0.0) for t in tickers_held}
            prices_df = self._fetch_period_close_prices(tickers_held, record_date, next_boundary)

            # 마크 기록: hold_and_mark는 execution-date MTM 마크(fresh closes)를 사용하고,
            # 그 외(OK/NO_ELIGIBLE)는 리밸런싱 직후 스냅샷을 그대로 사용한다.
            if screen_status == ScreenStatus.DATA_UNAVAILABLE:
                if prices_df.empty:
                    # total OHLCV 실패 → stale/flat-fill 마크 금지 (fail-fast)
                    raise RuntimeError(
                        f"DATA_UNAVAILABLE (hold_and_mark): {record_date} — 보유 종목 OHLCV 조회가 "
                        f"전체 실패했습니다 (tickers={tickers_held}). flat-fill 마크를 생성하지 않습니다."
                    )
                first_row = prices_df.iloc[0]
                uncovered = [
                    t for t in tickers_held
                    if t not in prices_df.columns or pd.isna(first_row.get(t))
                ]
                if uncovered:
                    print(f"  [MARK] ⚠ 마크일 가격 미조회(부분 실패), current_price fallback: {uncovered}")
                stock_val = sum(
                    shares_map.get(t, 0) * (
                        float(first_row[t]) if t in first_row.index and not pd.isna(first_row[t])
                        else fallback_prices.get(t, 0.0)
                    )
                    for t in tickers_held
                )
                mark_value = self.cash + stock_val
                recorded_dates.add(record_date)
                self.portfolio_history.append({
                    'date': record_date,
                    'portfolio_value': mark_value,
                    'cash': self.cash,
                    'stock_value': stock_val,
                    'num_holdings': len(self.portfolio),
                    'return': (mark_value - self.initial_capital) / self.initial_capital
                })
                print(f"  포트폴리오 가치: {mark_value:,.0f}원 ({(mark_value/self.initial_capital-1)*100:.2f}%)")
            else:
                portfolio_value = self.get_portfolio_value()
                # OK-path 리밸런싱 직후 스냅샷이 같은 날짜의 기존(일별) 기록보다 늦게 쌓이므로
                # calculate_performance의 drop_duplicates(keep='last')가 최신 스냅샷을 유지한다.
                recorded_dates.add(record_date)
                self.portfolio_history.append({
                    'date': record_date,
                    'portfolio_value': portfolio_value,
                    'cash': self.cash,
                    'stock_value': portfolio_value - self.cash,
                    'num_holdings': len(self.portfolio),
                    'return': (portfolio_value - self.initial_capital) / self.initial_capital
                })
                print(f"  포트폴리오 가치: {portfolio_value:,.0f}원 ({(portfolio_value/self.initial_capital-1)*100:.2f}%)")

            # 리밸런싱 구간 내 일별 포트폴리오 가치 기록 (MDD 계산 정확도 향상)
            # 일별 모니터링도 실제 매수일(record_date)부터 시작해 가짜 pre-purchase 데이터를 제거한다.
            # 첫 행(record_date 마크)은 recorded_dates 가드로 스킵된다.
            if len(self.portfolio) > 0 and not prices_df.empty:
                for dt_idx, row in prices_df.iterrows():
                    date_str = dt_idx.strftime('%Y-%m-%d') if hasattr(dt_idx, 'strftime') else str(dt_idx)[:10]
                    if date_str in recorded_dates:
                        continue
                    recorded_dates.add(date_str)
                    stock_val = sum(
                        shares_map.get(t, 0) * (
                            float(row[t]) if t in row.index and not pd.isna(row[t])
                            else fallback_prices.get(t, 0.0)
                        )
                        for t in tickers_held
                    )
                    total_val = self.cash + stock_val
                    self.portfolio_history.append({
                        'date': date_str,
                        'portfolio_value': total_val,
                        'cash': self.cash,
                        'stock_value': stock_val,
                        'num_holdings': len(self.portfolio),
                        'return': (total_val - self.initial_capital) / self.initial_capital
                    })

            self._log_timing('rebalance.total', time.perf_counter() - t_rebal_start)

        # 종료일 기준 최종 평가 추가 (리밸런싱일과 다를 수 있음)
        if len(self.portfolio_history) > 0:
            end_trading_date = self.selector.previous_trading_date(self.end_date)
            end_trading_date_fmt = datetime.strptime(end_trading_date, '%Y%m%d').strftime('%Y-%m-%d')
            last_recorded_date = self.portfolio_history[-1]['date']

            if end_trading_date_fmt != last_recorded_date:
                end_tickers = self.selector.get_market_tickers(end_trading_date_fmt)
                if len(end_tickers) > 0 and len(self.portfolio) > 0:
                    try:
                        # 종료일 가격만 업데이트 (종목 선정 아님) - selector의 Kiwoom 우선 경로 사용
                        end_df = self.selector._get_fundamental_and_cap(
                            end_trading_date_fmt.replace('-', ''),
                            markets=['KOSPI', 'KOSDAQ'],
                        )

                        if end_df is not None and not end_df.empty and 'ticker' in end_df.columns and 'close' in end_df.columns:
                            end_price_map = end_df.set_index('ticker')['close'].to_dict()
                            for ticker, position in self.portfolio.items():
                                if ticker in end_price_map and end_price_map[ticker] > 0:
                                    position['current_price'] = end_price_map[ticker]
                    except Exception:
                        pass

                final_value = self.get_portfolio_value()
                if end_trading_date_fmt not in recorded_dates:
                    recorded_dates.add(end_trading_date_fmt)
                    self.portfolio_history.append({
                        'date': end_trading_date_fmt,
                        'portfolio_value': final_value,
                        'cash': self.cash,
                        'stock_value': final_value - self.cash,
                        'num_holdings': len(self.portfolio),
                        'return': (final_value - self.initial_capital) / self.initial_capital
                    })

            self.selector.persist_caches()
            self._log_timing('backtest.total', time.perf_counter() - total_start)
        
        # Fail-fast 가드 (hold_and_mark 백스톱): 시도한 모든 리밸런싱이 DATA_UNAVAILABLE였다면
        # 거짓 성과 지표를 반환하는 대신 명시적으로 오류를 발생시킨다.
        # (strict 모드는 루프에서 첫 발생 시 즉시 raise하므로 여기까지 도달하지 않는다)
        if len(rebalance_dates) > 0 and len(self._data_missing_dates) >= len(rebalance_dates):
            raise RuntimeError(
                "모든 리밸런싱 시도가 데이터 조회 실패(DATA_UNAVAILABLE) 상태였습니다: "
                f"{self._data_missing_dates}. 거짓 수익률 지표를 생성하지 않습니다."
            )

        return self.calculate_performance()
    
    def calculate_performance(self):
        """성과 분석"""
        
        if len(self.portfolio_history) == 0:
            return None
        
        df = pd.DataFrame(self.portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)  # 일별 기록 삽입 후 순서 보장
        # 같은 날짜 중복 기록(경계/리밸런싱 재기록)은 마지막(최신, 리밸런싱 직후) 스냅샷을 유지한다.
        df = df.drop_duplicates(subset='date', keep='last')

        # 수익률 계산
        final_value = df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # 연수 계산
        years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
        
        # 연평균성장률(CAGR) 계산
        cagr = (final_value / self.initial_capital) ** (1/years) - 1 if years > 0 else 0
        
        # MDD 계산
        cummax = df['portfolio_value'].cummax()
        drawdown = (df['portfolio_value'] - cummax) / cummax
        mdd = drawdown.min()
        
        # 승률 계산 (월별 기준 — 일별 노이즈 제거)
        _monthly_vals = df.set_index('date')['portfolio_value'].resample('ME').last().dropna()
        _monthly_rets = _monthly_vals.pct_change().dropna()
        win_rate = ((_monthly_rets > 0).sum() / len(_monthly_rets)
                    if len(_monthly_rets) > 0 else 0)

        # 샤프 비율 (연환산, 무위험수익률 3.5%)
        # 공식: (CAGR - Rf) / (일별 변동성 × √252)
        # gap-spanning 수익률(date gap > 4일)은 일별 변동성 산정에서 제외 —
        # 결측 구간(데이터 부재/비영업일)이 연속 수익률처럼 변동성을 부풀리는 착시를 방지한다.
        _rf = 0.035
        _vol_series = df.set_index('date')['portfolio_value']
        _daily_rets = _vol_series.pct_change().dropna()
        _gap_days = _vol_series.index.to_series().diff().dt.days
        _gap_days = _gap_days.loc[_daily_rets.index]
        _daily_rets = _daily_rets[_gap_days <= _GAP_EXCLUSION_DAYS]
        _annual_vol = _daily_rets.std() * np.sqrt(252) if len(_daily_rets) > 1 else 0.0
        sharpe = (cagr - _rf) / _annual_vol if _annual_vol > 0 else 0

        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return * 100,
            'cagr_pct': cagr * 100,
            'mdd_pct': mdd * 100,
            'win_rate_pct': win_rate * 100,
            'sharpe_ratio': sharpe,
            'years': years,
            'requested_start_date': self.start_date,
            'requested_end_date': self.end_date,
            'actual_start_date': df['date'].iloc[0].strftime('%Y-%m-%d'),
            'actual_end_date': df['date'].iloc[-1].strftime('%Y-%m-%d'),
            'num_trades': len(self.trade_history),
            'data_missing_dates': list(self._data_missing_dates),
            'num_data_missing_days': len(self._data_missing_dates),
            # strict 실행은 첫 실패 시 즉시 raise되므로 여기 도달하는 degraded는
            # hold_and_mark 실행 중 실패일이 하나 이상 있었던 경우뿐이다.
            'degraded': self.data_unavailable_policy == 'hold_and_mark' and len(self._data_missing_dates) > 0,
            'portfolio_df': df,
            'trades_df': pd.DataFrame(self.trade_history)
        }
        
        return results
    
    def print_results(self, results):
        """결과 출력"""
        
        if results is None:
            print("백테스트 결과가 없습니다.")
            return
        
        print("\n" + "="*80)
        print("백테스트 결과")
        print("="*80)
        print(f"초기 자본:    {results['initial_capital']:>15,}원")
        print(f"최종 자산:    {results['final_value']:>15,.0f}원")
        print(f"총 수익률:    {results['total_return_pct']:>15.2f}%")
        print(f"CAGR:         {results['cagr_pct']:>15.2f}%")
        print(f"MDD:          {results['mdd_pct']:>15.2f}%")
        print(f"승률:         {results['win_rate_pct']:>15.2f}%")
        print(f"샤프 비율:    {results['sharpe_ratio']:>15.2f}")
        print(f"요청 기간:    {results['requested_start_date']} ~ {results['requested_end_date']}")
        print(f"실행 기간:    {results['actual_start_date']} ~ {results['actual_end_date']}")
        print(f"백테스트 기간: {results['years']:>14.2f}년")
        print(f"총 거래 횟수:  {results['num_trades']:>15}회")
        print("="*80)
        
        # 데이터 조회 실패 경고 (리밸런싱이 건너뛰어진 일수; strict 모드는 첫 실패 시 즉시
        # raise하므로 이 경고가 출력되는 경우는 hold_and_mark(degraded) 실행뿐이다)
        if results.get('num_data_missing_days', 0) > 0:
            print("\n" + "="*80)
            print(f"⚠ [DEGRADED] 데이터 조회 실패일: {results['num_data_missing_days']}일 — 해당 리밸런싱은 보유 유지 마크로 기록됨 (hold_and_mark)")
            print(f"  실패일: {', '.join(results.get('data_missing_dates', []))}")
            print("  ⚠ 결과 지표는 결측 구간을 보유 유지로 가정한 진단용 수치입니다. 샤프/변동성은 gap-exclusion 적용 후 산출됩니다.")
            print("="*80)
        
        # 연도별 수익률 (전년 말 포트폴리오 가치 대비 해당 연도 수익률)
        df = results['portfolio_df'].copy()
        df['year'] = df['date'].dt.year
        _yearly_vals = df.groupby('year')['portfolio_value'].last()
        _yearly_rets = _yearly_vals.pct_change() * 100
        # 첫 해는 초기자본 대비 계산
        _yearly_rets.iloc[0] = (_yearly_vals.iloc[0] / results['initial_capital'] - 1) * 100

        print("\n연도별 수익률:")
        print("-"*40)
        for year, ret in _yearly_rets.items():
            print(f"{year}: {ret:>10.2f}%")
        print("-"*40)


def main():
    """메인 실행 함수"""
    # 단일 실행: 사용자가 선택한 기준 적용
    import os
    os.makedirs('results', exist_ok=True)
    
    # 환경변수 기반 기본 설정 로드 (공통 설정 모델 사용)
    try:
        backtest_config = BacktestConfig.from_env()
    except ValueError:
        print("경고: .env 파일의 설정이 유효하지 않습니다. 기본값을 사용합니다.")
        backtest_config = BacktestConfig()

    print(
        f"로드된 설정: commission_fee_rate={backtest_config.commission_fee_rate*100:.2f}%, "
        f"tax_rate={backtest_config.tax_rate*100:.2f}%"
    )
    print(f"백테스트 기간: {backtest_config.start_date} ~ {backtest_config.end_date}")
    print(f"초기자본: {backtest_config.initial_capital:,}원")
    print(f"보유종목수: {backtest_config.num_stocks}개")

    # CLI 파서: CLI 인자 > 환경변수(.env 포함) > 기본값
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--rebalance-months', '-r', type=int, help='리밸런싱 주기(개월). CLI가 우선 적용됩니다')
    parser.add_argument('--rebalance-days', type=int, help='리밸런싱 주기(일). 설정되면 월 단위보다 우선 적용됩니다')
    args, _ = parser.parse_known_args()

    if args.rebalance_months is not None:
        backtest_rebalance_months = int(args.rebalance_months)
    else:
        reb_env = env_get('REBALANCE_MONTHS', fallback_keys=['REBALANCE_MONTHS', 'LIVE_REBALANCE_MONTHS'])
        if reb_env not in (None, ""):
            try:
                backtest_rebalance_months = int(reb_env)
            except Exception:
                backtest_rebalance_months = DEFAULT_REBALANCE_MONTHS
        else:
            backtest_rebalance_months = DEFAULT_REBALANCE_MONTHS

    if args.rebalance_days is not None:
        backtest_rebalance_days = int(args.rebalance_days)
    else:
        reb_days_env = env_get('REBALANCE_DAYS', fallback_keys=['REBALANCE_DAYS', 'LIVE_REBALANCE_DAYS'])
        if reb_days_env not in (None, ""):
            try:
                parsed_days = int(reb_days_env)
                backtest_rebalance_days = parsed_days if parsed_days > 0 else None
            except Exception:
                backtest_rebalance_days = None
        else:
            backtest_rebalance_days = None

    if backtest_rebalance_days is not None and backtest_rebalance_days > 0:
        rebalance_desc = f"{backtest_rebalance_days}d"
    else:
        rebalance_desc = f"{backtest_rebalance_months}m"
        
    # CLI/환경변수로 계산한 리밸런싱 값을 config에 반영
    backtest_config.rebalance_months = backtest_rebalance_months
    backtest_config.rebalance_days = backtest_rebalance_days

    if backtest_rebalance_days is not None and backtest_rebalance_days > 0:
        print(f"리밸런싱 주기: {backtest_rebalance_days}일")
    else:
        print(f"리밸런싱 주기: {backtest_rebalance_months}개월")

    # 변동성 타게팅 설정 출력
    if backtest_config.vol_target_enabled:
        print(
            f"[VOL-TARGET] 활성화: σ_target={backtest_config.vol_target_sigma*100:.0f}%, "
            f"lookback={backtest_config.vol_target_lookback}일, "
            f"min_ratio={backtest_config.vol_target_min_ratio:.0%}"
        )

    backtest = KoreaStockBacktest.from_config(backtest_config)

    results = backtest.run_backtest()
    if results:
        backtest.print_results(results)
        results['portfolio_df'].to_csv('results/backtest_portfolio.csv', index=False, encoding='utf-8-sig')
        results['trades_df'].to_csv('results/backtest_trades.csv', index=False, encoding='utf-8-sig')
        print("\nSaved: results/backtest_portfolio.csv, results/backtest_trades.csv")


if __name__ == "__main__":
    main()
