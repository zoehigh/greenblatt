from __future__ import annotations

import argparse
from utils.env import env_get
from collections import OrderedDict
import threading
import concurrent.futures
from datetime import datetime, timedelta
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import asyncio

KIWOOM_AVAILABLE = False

PYKRX_SESSION_ENABLED = False
_PYKRX_SESSION_INIT_ATTEMPTED = False
_PYKRX_SESSION_INIT_LOCK = threading.Lock()

def initialize_pykrx_session() -> bool:
    """런타임에 한 번만 pykrx 세션을 초기화합니다.

    임포트 시점이 아니라 런타임에서 호출하면 엔트리포인트가 `.env` 값을
    먼저 로드한 이후에 세션/로그인 설정을 시도하므로 안전합니다.
    """
    global PYKRX_SESSION_ENABLED, _PYKRX_SESSION_INIT_ATTEMPTED

    with _PYKRX_SESSION_INIT_LOCK:
        if _PYKRX_SESSION_INIT_ATTEMPTED:
            return PYKRX_SESSION_ENABLED
        _PYKRX_SESSION_INIT_ATTEMPTED = True

        # 선택적 헬퍼: 사용 불가하거나 실패하면 세션 패치를 건너뜁니다.
        try:
            from pykrx_session import enable_pykrx_session

            enable_pykrx_session()
            PYKRX_SESSION_ENABLED = True
        except Exception:
            PYKRX_SESSION_ENABLED = False
        return PYKRX_SESSION_ENABLED

try:
    from pykrx import stock
    LIBRARIES_AVAILABLE = True
except ImportError:
    LIBRARIES_AVAILABLE = False

# pykrx에서 사용하는 requests에 대해 합리적인 기본 타임아웃을 강제하여
# 장기간 블로킹되는 호출을 방지합니다.
try:
    import requests
    import os

    _orig_session_request = requests.sessions.Session.request

    def _session_request_with_timeout(self, method, url, **kwargs):
        if 'timeout' not in kwargs:
            try:
                kwargs['timeout'] = float(env_get('PYKRX_REQUEST_TIMEOUT', default='6.0'))
            except Exception:
                kwargs['timeout'] = 6.0
        return _orig_session_request(self, method, url, **kwargs)

    requests.sessions.Session.request = _session_request_with_timeout
except Exception:
    pass


class KoreaStockSelector:
    """백테스트/실전 공용 종목선정 엔진."""

    def __init__(
        self,
        *,
        num_stocks: int = 40,
        strategy_mode: str = "mixed",
        mixed_filter_profile: str = "large_cap",
        kosdaq_target_ratio: float | None = None,
        momentum_enabled: bool = True,
        momentum_months: int = 3,
        momentum_weight: float = 0.6,
        momentum_filter_enabled: bool = False,
        large_cap_min_mcap: float | None = None,
        sector_max_stocks: int | None = None,
        cache_dir: str = "results/cache",
        timing_enabled: bool = True,
        fundamental_cache_format: str = "parquet",
        fundamental_cache_max_entries: int = 16,
        fundamental_source: str | None = None,
    ) -> None:
        initialize_pykrx_session()

        self.num_stocks = num_stocks
        self.strategy_mode = strategy_mode
        self.mixed_filter_profile = mixed_filter_profile
        self.kosdaq_target_ratio = kosdaq_target_ratio
        self.momentum_enabled = momentum_enabled
        self.momentum_months = momentum_months
        self.momentum_weight = momentum_weight
        self.momentum_filter_enabled = momentum_filter_enabled
        self.large_cap_min_mcap = large_cap_min_mcap
        self.sector_max_stocks = sector_max_stocks
        self.cache_dir = cache_dir
        self.timing_enabled = timing_enabled
        self.fundamental_cache_format = fundamental_cache_format
        self.fundamental_cache_max_entries = max(1, int(fundamental_cache_max_entries))
        source_from_env = env_get("FUNDAMENTAL_SOURCE", fallback_keys=["LIVE_FUNDAMENTAL_SOURCE", "BACKTEST_FUNDAMENTAL_SOURCE"], default="pykrx")
        self.fundamental_source = str(fundamental_source or source_from_env).strip().lower()
        if self.fundamental_source not in {"auto", "kiwoom", "pykrx"}:
            print(f"[FUND] invalid fundamental_source={self.fundamental_source}, fallback=pykrx")
            self.fundamental_source = "pykrx"
        self.cache_version = {
            "fundamental_cache_v": 2,
            "momentum_cache_v": 1,
            "strategy_mode": self.strategy_mode,
            "momentum_months": self.momentum_months,
            "cache_format": self.fundamental_cache_format,
            "fundamental_source": self.fundamental_source,
        }

        self.industry_cache: dict[str, str] = {}
        self.momentum_cache: dict[str, float] = {}
        self.price_cache: dict[str, float] = {}
        self.ticker_list_cache: dict[str, list[str]] = {}
        self.fundamental_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

        self._load_caches()

    def _should_try_kiwoom_fundamental(self, normalized_date: str) -> bool:
        if self.fundamental_source == "pykrx":
            return False
        if self.fundamental_source == "kiwoom":
            return self._can_use_kiwoom_for_date(normalized_date)
        return self._can_use_kiwoom_for_date(normalized_date)

    def _should_try_pykrx_fundamental(self) -> bool:
        # kiwoom 모드에서도 장애 복원력을 위해 pykrx fallback은 허용
        return True

    def _cache_paths(self):
        return {
            "industry": os.path.join(self.cache_dir, "industry_cache.json"),
            "momentum": os.path.join(self.cache_dir, "momentum_cache.json"),
            "price": os.path.join(self.cache_dir, "price_cache.json"),
            "ticker_list": os.path.join(self.cache_dir, "ticker_list_cache.json"),
            "fundamentals": os.path.join(self.cache_dir, "fundamentals"),
            "meta": os.path.join(self.cache_dir, "cache_meta.json"),
        }

    def _validate_cache_version(self, loaded_meta):
        if not isinstance(loaded_meta, dict):
            return False
        for key, value in self.cache_version.items():
            if loaded_meta.get(key) != value:
                return False
        return True

    def _purge_fundamental_disk_cache(self):
        paths = self._cache_paths()
        fundamentals_dir = paths["fundamentals"]
        if not os.path.isdir(fundamentals_dir):
            return
        for file_name in os.listdir(fundamentals_dir):
            if file_name.endswith(".parquet") or file_name.endswith(".csv"):
                try:
                    os.remove(os.path.join(fundamentals_dir, file_name))
                except Exception:
                    pass

    def _fundamental_cache_file(self, date_str, market):
        paths = self._cache_paths()
        fundamentals_dir = paths["fundamentals"]
        preferred_ext = ".parquet" if self.fundamental_cache_format == "parquet" else ".csv"
        preferred = os.path.join(fundamentals_dir, f"{date_str}_{market}{preferred_ext}")
        alternate_ext = ".csv" if preferred_ext == ".parquet" else ".parquet"
        alternate = os.path.join(fundamentals_dir, f"{date_str}_{market}{alternate_ext}")
        return preferred, alternate

    def _remove_fundamental_cache_file(self, date_str, market):
        preferred, alternate = self._fundamental_cache_file(date_str, market)
        for path in [preferred, alternate]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    def _is_valid_fundamental_frame(self, df: pd.DataFrame) -> bool:
        required_cols = ["PER", "PBR", "market_cap"]
        if df is None or df.empty:
            return False
        if any(col not in df.columns for col in required_cols):
            return False

        try:
            per = pd.to_numeric(df["PER"], errors="coerce")
            pbr = pd.to_numeric(df["PBR"], errors="coerce")
            mcap = pd.to_numeric(df["market_cap"], errors="coerce")
        except Exception:
            return False

        # 비거래일/비정상 스냅샷에서 PER/PBR/시총이 전부 0으로 들어오는 경우를 차단합니다.
        return bool((per > 0).any() and (pbr > 0).any() and (mcap > 0).any())

    def _load_fundamental_frame(self, date_str, market):
        preferred, alternate = self._fundamental_cache_file(date_str, market)
        for path in [preferred, alternate]:
            if not os.path.exists(path):
                continue
            try:
                if path.endswith(".parquet"):
                    return pd.read_parquet(path)
                return pd.read_csv(path)
            except Exception:
                continue
        return None

    def _save_fundamental_frame(self, date_str, market, df):
        preferred, _ = self._fundamental_cache_file(date_str, market)
        try:
            if preferred.endswith(".parquet"):
                df.to_parquet(preferred, index=False)
            else:
                df.to_csv(preferred, index=False, encoding="utf-8-sig")
            return
        except Exception:
            pass

        fallback = preferred.replace(".parquet", ".csv")
        try:
            df.to_csv(fallback, index=False, encoding="utf-8-sig")
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
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass
        return {}

    def _save_json_cache(self, path, data):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            print(f"캐시 저장 실패 ({path}): {e}")

    def _load_caches(self):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            paths = self._cache_paths()
            os.makedirs(paths["fundamentals"], exist_ok=True)

            loaded_meta = self._load_json_cache(paths["meta"])
            if not self._validate_cache_version(loaded_meta):
                print("[CACHE] cache version mismatch detected, invalidating incompatible caches")
                self.momentum_cache = {}
                self.fundamental_cache = OrderedDict()
                self._purge_fundamental_disk_cache()

            self.industry_cache = self._load_json_cache(paths["industry"])
            if len(self.momentum_cache) == 0:
                self.momentum_cache = self._load_json_cache(paths["momentum"])
            self.price_cache = self._load_json_cache(paths["price"])
            raw_ticker_cache = self._load_json_cache(paths["ticker_list"])
            normalized_ticker_cache: dict[str, list[str]] = {}
            for key, value in raw_ticker_cache.items():
                if isinstance(value, list):
                    normalized_ticker_cache[str(key)] = [str(item) for item in value if str(item).strip()]
            self.ticker_list_cache = normalized_ticker_cache
            print(
                f"[CACHE] loaded: industry={len(self.industry_cache):,}, "
                f"momentum={len(self.momentum_cache):,}, "
                f"price={len(self.price_cache):,}, ticker_list={len(self.ticker_list_cache):,}"
            )
        except Exception as e:
            print(f"[CACHE] load failed: {e}")
            self.industry_cache = {}
            self.momentum_cache = {}
            self.price_cache = {}
            self.ticker_list_cache = {}
            self.fundamental_cache = OrderedDict()

    def persist_caches(self):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            paths = self._cache_paths()
            os.makedirs(paths["fundamentals"], exist_ok=True)
            self._save_json_cache(paths["industry"], self.industry_cache)
            self._save_json_cache(paths["momentum"], self.momentum_cache)
            self._save_json_cache(paths["price"], self.price_cache)
            self._save_json_cache(paths["ticker_list"], self.ticker_list_cache)
            self._save_json_cache(paths["meta"], self.cache_version)

            for cache_key, df in self.fundamental_cache.items():
                try:
                    parts = cache_key.split("|")
                    if len(parts) == 2:
                        date_str, market = parts
                        self._save_fundamental_frame(date_str, market, df)
                except Exception:
                    pass
            print(
                f"[CACHE] saved: industry={len(self.industry_cache):,}, "
                f"momentum={len(self.momentum_cache):,}, "
                f"ticker_list={len(self.ticker_list_cache):,}, "
                f"fundamental={len(self.fundamental_cache):,}"
            )
        except Exception as e:
            print(f"[CACHE] save failed: {e}")

    def _normalize_date_yyyymmdd(self, date_str: str) -> str:
        return str(date_str).replace("-", "")

    def _is_today(self, date_str: str) -> bool:
        return self._normalize_date_yyyymmdd(date_str) == datetime.now().strftime("%Y%m%d")

    def _allow_kiwoom_date_proxy(self) -> bool:
        return env_get("KIWOOM_ALLOW_DATE_PROXY", default="false").lower() in {"1", "true", "yes", "y"}

    def _can_use_kiwoom_for_date(self, date_str: str) -> bool:
        if self._is_today(date_str):
            return True
        return self._allow_kiwoom_date_proxy()

    def _ticker_cache_key(self, date_str: str, market: str) -> str:
        return f"{self._normalize_date_yyyymmdd(date_str)}|{str(market).upper()}"

    def _run_async_safely(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result = {}
        error = {}

        def _target():
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                result["value"] = loop.run_until_complete(coro)
            except Exception as e:
                error["value"] = e
            finally:
                loop.close()

        t = threading.Thread(target=_target)
        t.start()
        t.join()

        if "value" in error:
            raise error["value"]
        return result.get("value")

    def _kiwoom_market_code(self, market: str) -> str:
        return "10" if str(market).upper() == "KOSDAQ" else "0"

    def _prefilter_tickers_by_liquidity_and_cap(self, tickers: list[str], date_str: str, market: str) -> list[str]:
        """Reduce Kiwoom per-ticker calls by prefiltering on market cap/liquidity.

        Uses pykrx market cap snapshot once per market/date, then keeps top candidates.
        """
        if len(tickers) <= 1:
            return tickers

        enabled = env_get("KIWOOM_PREFILTER_ENABLED", default="true").lower() in {"1", "true", "yes", "y"}
        if not enabled or not LIBRARIES_AVAILABLE:
            return tickers

        try:
            target_count = max(1, int(env_get("KIWOOM_PREFILTER_TARGET", default="500")))
            min_mcap = float(env_get("KIWOOM_PREFILTER_MIN_MCAP", default="50000000000"))
            min_tvalue = float(env_get("KIWOOM_PREFILTER_MIN_TRADING_VALUE", default="0"))
        except Exception:
            target_count = 500
            min_mcap = 5e10
            min_tvalue = 0.0

        t_start = time.perf_counter()
        try:
            requested_date = self._normalize_date_yyyymmdd(date_str)
            candidate_dates = [requested_date]
            try:
                prev_date = stock.get_nearest_business_day_in_a_week(date=requested_date, prev=True)
                prev_date = self._normalize_date_yyyymmdd(prev_date)
                if prev_date not in candidate_dates:
                    candidate_dates.append(prev_date)
            except Exception:
                pass

            # 주말/공휴일/네트워크 장애에 대한 폴백: 최근 영업일을 탐색하여 보완합니다.
            try:
                base_dt = datetime.strptime(requested_date, "%Y%m%d")
                for i in range(1, 8):
                    d = (base_dt - timedelta(days=i)).strftime("%Y%m%d")
                    if d not in candidate_dates:
                        candidate_dates.append(d)
            except Exception:
                pass

            today = datetime.now().strftime("%Y%m%d")
            if today not in candidate_dates:
                candidate_dates.append(today)

            df_cap = None
            used_date = None
            for candidate_date in candidate_dates:
                try:
                    df_try = stock.get_market_cap_by_ticker(candidate_date, market=market)
                    if df_try is not None and not df_try.empty:
                        df_cap = df_try
                        used_date = candidate_date
                        break
                except Exception:
                    continue

            if df_cap is None or df_cap.empty:
                print(
                    f"  [FUND][PREFILTER] skip: market={market}, reason=no_market_cap_snapshot, "
                    f"requested_date={requested_date}, candidates={candidate_dates}"
                )
                return tickers

            df_cap = df_cap.reset_index().rename(
                columns={"티커": "ticker", "시가총액": "market_cap", "거래대금": "trading_value", "거래량": "volume"}
            )
            if "ticker" not in df_cap.columns:
                return tickers

            ticker_set = set(tickers)
            df_cap = df_cap[df_cap["ticker"].astype(str).isin(ticker_set)].copy()
            if df_cap.empty:
                return tickers

            if "market_cap" in df_cap.columns:
                df_cap["market_cap"] = pd.to_numeric(df_cap["market_cap"], errors="coerce")
            else:
                df_cap["market_cap"] = np.nan

            if "trading_value" in df_cap.columns:
                df_cap["trading_value"] = pd.to_numeric(df_cap["trading_value"], errors="coerce")
            else:
                df_cap["trading_value"] = np.nan

            filtered = df_cap
            if min_mcap > 0:
                filtered = filtered[filtered["market_cap"] >= min_mcap]
            if min_tvalue > 0:
                filtered = filtered[filtered["trading_value"] >= min_tvalue]

            if filtered.empty:
                print(
                    f"  [FUND][PREFILTER] empty after thresholds: market={market}, "
                    f"before={len(tickers)}, min_mcap={min_mcap:.0f}, min_tvalue={min_tvalue:.0f}"
                )
                return tickers

            filtered = filtered.sort_values(["trading_value", "market_cap"], ascending=[False, False], na_position="last")
            reduced = filtered["ticker"].astype(str).head(target_count).tolist()
            if len(reduced) == 0:
                return tickers

            self._log_timing(
                "fund.prefilter",
                time.perf_counter() - t_start,
                extra=(
                    f"market={market}, before={len(tickers)}, after={len(reduced)}, "
                    f"target={target_count}, source_date={used_date}"
                ),
            )
            return reduced
        except Exception as e:
            print(f"  [FUND][PREFILTER] failed: market={market}, error={type(e).__name__}: {e}")
            return tickers

    def _fetch_kiwoom_ticker_list_sync(self, market: str, date_str: str) -> list[str]:
        normalized_date = self._normalize_date_yyyymmdd(date_str)
        cache_key = self._ticker_cache_key(normalized_date, market)
        if cache_key in self.ticker_list_cache:
            return list(self.ticker_list_cache.get(cache_key, []))

        if not self._can_use_kiwoom_for_date(normalized_date):
            return []

        async def _inner():
            from live_trading.kiwoom_adapter import KiwoomBrokerAdapter
            from live_trading.config import LiveTradingConfig

            config = LiveTradingConfig.from_env()
            adapter = KiwoomBrokerAdapter(config)
            await adapter.connect()
            try:
                list_endpoint = env_get("KIWOOM_STOCK_LIST_ENDPOINT", default="/api/dostk/stkinfo")
                list_api_id = env_get("KIWOOM_STOCK_LIST_API_ID", default="ka10099")
                max_pages = int(env_get("KIWOOM_STOCK_LIST_MAX_PAGES", default="50"))

                body = await adapter.request_endpoint_paginated(
                    endpoint=list_endpoint,
                    api_id=list_api_id,
                    data={"mrkt_tp": self._kiwoom_market_code(market)},
                    list_key="list",
                    max_pages=max_pages,
                )

                raw_list = body.get("list") if isinstance(body, dict) else None
                if not isinstance(raw_list, list):
                    return []

                tickers = []
                for item in raw_list:
                    if not isinstance(item, dict):
                        continue
                    code = str(item.get("code") or "").strip()
                    if code:
                        tickers.append(code)
                return list(dict.fromkeys(tickers))
            finally:
                await adapter.close()

        tickers = self._run_async_safely(_inner())
        if isinstance(tickers, list) and len(tickers) > 0:
            self.ticker_list_cache[cache_key] = list(tickers)
            return list(tickers)
        return []

    def _fetch_kiwoom_fundamental_and_cap_sync(self, date_str: str, market: str) -> pd.DataFrame:
        normalized_date = self._normalize_date_yyyymmdd(date_str)
        if not self._can_use_kiwoom_for_date(normalized_date):
            print(f"  [FUND][KIWOOM] skip: market={market}, date={normalized_date}, reason=date_not_allowed")
            return pd.DataFrame()

        async def _inner():
            from live_trading.kiwoom_adapter import KiwoomBrokerAdapter
            from live_trading.config import LiveTradingConfig

            config = LiveTradingConfig.from_env()
            adapter = KiwoomBrokerAdapter(config)
            await adapter.connect()
            try:
                print(f"  [FUND][KIWOOM] start: market={market}, date={normalized_date}")
                cache_key = self._ticker_cache_key(normalized_date, market)
                tickers = list(self.ticker_list_cache.get(cache_key, []))
                if len(tickers) > 0:
                    print(f"  [FUND][KIWOOM] ticker cache hit: market={market}, count={len(tickers)}")
                if len(tickers) == 0:
                    list_endpoint = env_get("KIWOOM_STOCK_LIST_ENDPOINT", default="/api/dostk/stkinfo")
                    list_api_id = env_get("KIWOOM_STOCK_LIST_API_ID", default="ka10099")
                    max_pages = int(env_get("KIWOOM_STOCK_LIST_MAX_PAGES", default="50"))

                    body = await adapter.request_endpoint_paginated(
                        endpoint=list_endpoint,
                        api_id=list_api_id,
                        data={"mrkt_tp": self._kiwoom_market_code(market)},
                        list_key="list",
                        max_pages=max_pages,
                    )

                    raw_list = body.get("list") if isinstance(body, dict) else None
                    if isinstance(raw_list, list):
                        tickers = [
                            str(item.get("code") or "").strip()
                            for item in raw_list
                            if isinstance(item, dict) and str(item.get("code") or "").strip()
                        ]
                        tickers = list(dict.fromkeys(tickers))
                        if len(tickers) > 0:
                            self.ticker_list_cache[cache_key] = list(tickers)
                    print(
                        f"  [FUND][KIWOOM] ticker fetch: market={market}, endpoint={list_endpoint}, "
                        f"count={len(tickers)}"
                    )

                if len(tickers) == 0:
                    print(f"  [FUND][KIWOOM] empty ticker list: market={market}, date={normalized_date}")
                    return pd.DataFrame()

                prefilter_before = len(tickers)
                tickers = self._prefilter_tickers_by_liquidity_and_cap(tickers, normalized_date, market)
                if len(tickers) != prefilter_before:
                    print(
                        f"  [FUND][KIWOOM] prefilter applied: market={market}, "
                        f"before={prefilter_before}, after={len(tickers)}"
                    )
                else:
                    print(
                        f"  [FUND][KIWOOM] prefilter no-change: market={market}, "
                        f"count={len(tickers)}"
                    )

                max_count = int(env_get("KIWOOM_FUND_MAX", default="0"))
                if max_count <= 0:
                    try:
                        prefilter_enabled = env_get("KIWOOM_PREFILTER_ENABLED", default="true").lower() in {"1", "true", "yes", "y"}
                        prefilter_target = max(1, int(env_get("KIWOOM_PREFILTER_TARGET", default="500")))
                    except Exception:
                        prefilter_enabled = True
                        prefilter_target = 500
                    # 폴백 상한: prefilter 데이터가 없더라도 요청량을 제한합니다.
                    if prefilter_enabled and len(tickers) > prefilter_target:
                        tickers = tickers[:prefilter_target]
                        print(
                            f"  [FUND][KIWOOM] prefilter hard-cap: market={market}, "
                            f"capped={len(tickers)}"
                        )
                if max_count > 0:
                    tickers = tickers[:max_count]
                    print(f"  [FUND][KIWOOM] ticker capped: market={market}, max_count={max_count}")

                concurrency = max(1, int(env_get("KIWOOM_FUND_CONCURRENCY", default="3")))
                sem = asyncio.Semaphore(concurrency)
                fetch_error_samples: list[str] = []

                async def _fetch_one(ticker: str):
                    async with sem:
                        try:
                            fam = await adapter.get_fundamental_by_ticker(ticker)
                            if not isinstance(fam, dict):
                                return None

                            close = fam.get("cur_prc")
                            if close is None:
                                close = fam.get("open_pric")

                            return {
                                "ticker": ticker,
                                "close": close,
                                "market_cap": fam.get("market_cap") or fam.get("mac"),
                                "PER": fam.get("per"),
                                "EPS": fam.get("eps"),
                                "ROE": fam.get("roe"),
                                "PBR": fam.get("pbr"),
                                "BPS": fam.get("bps"),
                                "DIV": fam.get("div"),
                                "market": market,
                            }
                        except Exception as e:
                            if len(fetch_error_samples) < 5:
                                fetch_error_samples.append(f"{ticker}:{type(e).__name__}:{e}")
                            return None

                tasks = [asyncio.create_task(_fetch_one(ticker)) for ticker in tickers]
                rows = [row for row in await asyncio.gather(*tasks) if row is not None]
                fail_count = max(0, len(tickers) - len(rows))
                # record Kiwoom-only stats before any pykrx 보완
                kiwoom_requested = len(tickers)
                kiwoom_success = len(rows)
                kiwoom_fail = fail_count
                print(
                    f"  [FUND][KIWOOM] fundamental fetch done: market={market}, "
                    f"requested={len(tickers)}, success={len(rows)}, fail={fail_count}, concurrency={concurrency}"
                )
                if len(fetch_error_samples) > 0:
                    print(f"  [FUND][KIWOOM] sample errors: {fetch_error_samples}")
                if len(rows) == 0:
                    print(f"  [FUND][KIWOOM] empty fundamentals: market={market}, date={normalized_date}")
                    return pd.DataFrame()

                # Kiwoom 응답으로부터 초기 DataFrame을 구성합니다
                df_ki = pd.DataFrame(rows)

                # If some tickers failed from Kiwoom, attempt per-ticker pykrx 보완 (only for missing tickers)
                try:
                    success_tickers = set(df_ki["ticker"].astype(str)) if not df_ki.empty and "ticker" in df_ki.columns else set()
                    missing = [t for t in tickers if str(t) not in success_tickers]
                except Exception:
                    missing = list(tickers)

                added = 0
                added_tickers: list[str] = []
                if missing and LIBRARIES_AVAILABLE:
                    try:
                        df_fund_mkt, fund_date = self._safe_pykrx_fundamental(normalized_date, market)
                        df_cap_mkt, cap_date = self._safe_pykrx_cap(normalized_date, market)
                        if df_fund_mkt is not None and df_cap_mkt is not None and not df_fund_mkt.empty and not df_cap_mkt.empty:
                            # 다른 경로와 동일하게 pykrx 프레임을 정규화하고 병합합니다
                            df_f = df_fund_mkt.reset_index().rename(columns={"티커": "ticker"}) if "티커" in df_fund_mkt.columns else df_fund_mkt.reset_index().rename(columns={df_fund_mkt.index.name or "index": "ticker"})
                            df_c = df_cap_mkt.reset_index().rename(columns={"티커": "ticker", "종가": "close", "시가총액": "market_cap"}) if "티커" in df_cap_mkt.columns or "종가" in df_cap_mkt.columns else df_cap_mkt.reset_index().rename(columns={df_cap_mkt.index.name or "index": "ticker"})
                            merged_py = pd.merge(df_f, df_c[["ticker", "close", "market_cap"]], on="ticker", how="inner")
                            merged_py["market"] = market
                            # 누락된 티커만 선택합니다
                            merged_missing = merged_py[merged_py["ticker"].astype(str).isin([str(x) for x in missing])]
                            if not merged_missing.empty:
                                # Kiwoom 응답과 동일한 dict 형태로 행을 변환합니다
                                added = 0
                                added_tickers: list[str] = []
                                for _, r in merged_missing.iterrows():
                                    try:
                                        row = {
                                            "ticker": str(r.get("ticker")),
                                            "close": float(r.get("close")) if r.get("close") is not None else None,
                                            "market_cap": float(r.get("market_cap")) if r.get("market_cap") is not None else None,
                                            "PER": float(r.get("PER")) if r.get("PER") is not None else None,
                                            "EPS": float(r.get("EPS")) if r.get("EPS") is not None else None,
                                            "ROE": float(r.get("ROE")) if r.get("ROE") is not None else None,
                                            "PBR": float(r.get("PBR")) if r.get("PBR") is not None else None,
                                            "BPS": float(r.get("BPS")) if r.get("BPS") is not None else None,
                                            "DIV": float(r.get("DIV")) if r.get("DIV") is not None else None,
                                            "market": market,
                                        }
                                        rows.append(row)
                                        added += 1
                                        added_tickers.append(str(r.get("ticker")))
                                    except Exception:
                                        continue
                                if added > 0:
                                    try:
                                        print(
                                            f"  [FUND][PYKRX] 보완 적용: market={market}, added={added}, missing_before={len(missing)}, added_tickers={added_tickers}"
                                        )
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                for col in ["close", "market_cap", "PER", "EPS", "ROE", "PBR", "BPS", "DIV"]:
                    if col in df_ki.columns:
                        df_ki[col] = pd.to_numeric(df_ki[col], errors="coerce")
                # if we appended pykrx 보완 rows, rebuild dataframe to include them
                try:
                    if len(rows) != len(df_ki):
                        df_ki = pd.DataFrame(rows)
                        for col in ["close", "market_cap", "PER", "EPS", "ROE", "PBR", "BPS", "DIV"]:
                            if col in df_ki.columns:
                                df_ki[col] = pd.to_numeric(df_ki[col], errors="coerce")
                except Exception:
                    pass

                # final success/fail after possible pykrx 보완
                try:
                    final_success = len(df_ki)
                    final_fail = max(0, kiwoom_requested - final_success)
                except Exception:
                    final_success = len(df_ki) if df_ki is not None else 0
                    final_fail = max(0, len(tickers) - final_success)

                print(f"  [FUND][KIWOOM] dataframe: market={market}, shape={df_ki.shape}, cols={list(df_ki.columns)}")
                try:
                    print(
                        f"  [FUND][SUMMARY] market={market}, requested={kiwoom_requested}, kiwoom_success={kiwoom_success}, kiwoom_fail={kiwoom_fail}, pykrx_added={added}, final_success={final_success}, final_fail={final_fail}"
                    )
                except Exception:
                    pass

                return df_ki
            finally:
                await adapter.close()

        return self._run_async_safely(_inner())

    def _build_pykrx_candidate_dates(self, date_str: str) -> list[str]:
        requested_date = self._normalize_date_yyyymmdd(date_str)
        candidates = [requested_date]

        try:
            prev_date = stock.get_nearest_business_day_in_a_week(date=requested_date, prev=True)
            prev_date = self._normalize_date_yyyymmdd(prev_date)
            if prev_date not in candidates:
                candidates.append(prev_date)
        except Exception:
            pass

        try:
            base_dt = datetime.strptime(requested_date, "%Y%m%d")
            for i in range(1, 8):
                d = (base_dt - timedelta(days=i)).strftime("%Y%m%d")
                if d not in candidates:
                    candidates.append(d)
        except Exception:
            pass

        return candidates

    def _safe_pykrx_fundamental(self, date_str: str, market: str):
        required_cols = {"BPS", "PER", "PBR", "EPS"}
        for candidate_date in self._build_pykrx_candidate_dates(date_str):
            try:
                df_fund = stock.get_market_fundamental_by_ticker(candidate_date, market=market)
            except KeyError as e:
                print(
                    f"  [FUND][PYKRX] fundamental key error: market={market}, "
                    f"date={candidate_date}, error={e}"
                )
                continue
            except Exception as e:
                print(
                    f"  [FUND][PYKRX] fundamental fetch failed: market={market}, "
                    f"date={candidate_date}, error={type(e).__name__}: {e}"
                )
                continue

            if df_fund is None or df_fund.empty:
                continue

            cols = set(str(c) for c in df_fund.columns)
            if not required_cols.issubset(cols):
                print(
                    f"  [FUND][PYKRX] fundamental schema mismatch: market={market}, "
                    f"date={candidate_date}, cols={list(df_fund.columns)}"
                )
                continue

            try:
                per = pd.to_numeric(df_fund["PER"], errors="coerce")
                pbr = pd.to_numeric(df_fund["PBR"], errors="coerce")
                if int(((per > 0) & (pbr > 0)).sum()) == 0:
                    print(
                        f"  [FUND][PYKRX] fundamental degenerate snapshot: market={market}, "
                        f"date={candidate_date}, reason=no_positive_per_pbr"
                    )
                    continue
            except Exception:
                continue
            return df_fund, candidate_date

        return None, None

    def _safe_pykrx_cap(self, date_str: str, market: str):
        required_cols = {"종가", "시가총액"}
        for candidate_date in self._build_pykrx_candidate_dates(date_str):
            try:
                df_cap = stock.get_market_cap_by_ticker(candidate_date, market=market)
            except Exception as e:
                print(
                    f"  [FUND][PYKRX] cap fetch failed: market={market}, "
                    f"date={candidate_date}, error={type(e).__name__}: {e}"
                )
                continue

            if df_cap is None or df_cap.empty:
                continue

            cols = set(str(c) for c in df_cap.columns)
            if not required_cols.issubset(cols):
                print(
                    f"  [FUND][PYKRX] cap schema mismatch: market={market}, "
                    f"date={candidate_date}, cols={list(df_cap.columns)}"
                )
                continue

            try:
                mcap = pd.to_numeric(df_cap["시가총액"], errors="coerce")
                if int((mcap > 0).sum()) == 0:
                    print(
                        f"  [FUND][PYKRX] cap degenerate snapshot: market={market}, "
                        f"date={candidate_date}, reason=no_positive_market_cap"
                    )
                    continue
            except Exception:
                continue
            return df_cap, candidate_date

        return None, None

    def _get_fundamental_and_cap(self, date_str, markets=["KOSPI", "KOSDAQ"]):
        normalized_date = self._normalize_date_yyyymmdd(date_str)
        t_start = time.perf_counter()
        cache_hits = 0
        cache_misses = 0
        dfs_merged = []

        if self.timing_enabled:
            print(
                f"  [FUND] source policy: source={self.fundamental_source}, "
                f"date={normalized_date}, try_kiwoom={self._should_try_kiwoom_fundamental(normalized_date)}"
            )

        for market in markets:
            cache_key = f"{normalized_date}|{market}"

            if cache_key in self.fundamental_cache:
                df_cached = self.fundamental_cache[cache_key].copy()
                if not self._is_valid_fundamental_frame(df_cached):
                    print(f"  [FUND][CACHE] invalid in-memory snapshot ignored: key={cache_key}")
                    try:
                        del self.fundamental_cache[cache_key]
                    except Exception:
                        pass
                    df_cached = None
                if df_cached is None:
                    pass
                else:
                    self.fundamental_cache.move_to_end(cache_key)
                    if "market" not in df_cached.columns:
                        df_cached["market"] = market
                    dfs_merged.append(df_cached)
                    cache_hits += 1
                    continue

            df_disk = self._load_fundamental_frame(normalized_date, market)
            if df_disk is not None:
                if not self._is_valid_fundamental_frame(df_disk):
                    print(f"  [FUND][CACHE] invalid disk snapshot removed: key={cache_key}")
                    self._remove_fundamental_cache_file(normalized_date, market)
                else:
                    if "market" not in df_disk.columns:
                        df_disk["market"] = market
                    self._set_fundamental_cache_lru(cache_key, df_disk)
                    dfs_merged.append(df_disk)
                    cache_hits += 1
                    continue

            if (not self._is_today(normalized_date)) and self._allow_kiwoom_date_proxy():
                print(
                    f"  [FUND] {market} date proxy enabled: requested={normalized_date}, "
                    f"kiwoom-current snapshot will be used"
                )

            try_kiwoom = self._should_try_kiwoom_fundamental(normalized_date)
            if try_kiwoom:
                try:
                    df_ki = self._fetch_kiwoom_fundamental_and_cap_sync(date_str=normalized_date, market=market)
                    if df_ki is not None and not df_ki.empty:
                        self._set_fundamental_cache_lru(cache_key, df_ki)
                        dfs_merged.append(df_ki)
                        cache_misses += 1
                        print(f"  [FUND] {market} via Kiwoom: {len(df_ki)} rows")
                        continue
                    print(f"  [FUND] {market} Kiwoom returned empty, fallback to pykrx")
                except Exception as e:
                    print(f"  [FUND] {market} Kiwoom fetch failed, fallback to pykrx: {type(e).__name__}: {e}")
                    print(traceback.format_exc())
            else:
                print(f"  [FUND] {market} skip Kiwoom by source policy: source={self.fundamental_source}")

            if not self._should_try_pykrx_fundamental():
                continue

            try:
                print(f"  [FUND][PYKRX] request start: market={market}, date={normalized_date}")
                df_fund_mkt, fund_date = self._safe_pykrx_fundamental(normalized_date, market)
                print(
                    f"  [FUND][PYKRX] fundamental raw: market={market}, "
                    f"source_date={fund_date}, "
                    f"shape={None if df_fund_mkt is None else df_fund_mkt.shape}, "
                    f"cols={None if df_fund_mkt is None else list(df_fund_mkt.columns)}"
                )
                df_cap_mkt, cap_date = self._safe_pykrx_cap(normalized_date, market)
                print(
                    f"  [FUND][PYKRX] cap raw: market={market}, "
                    f"source_date={cap_date}, "
                    f"shape={None if df_cap_mkt is None else df_cap_mkt.shape}, "
                    f"cols={None if df_cap_mkt is None else list(df_cap_mkt.columns)}"
                )

                # 디버그: 반환된 컬럼을 출력하여 스키마/포맷 변경을 진단합니다
                try:
                    fund_cols = list(df_fund_mkt.columns) if df_fund_mkt is not None else None
                except Exception:
                    fund_cols = None
                try:
                    cap_cols = list(df_cap_mkt.columns) if df_cap_mkt is not None else None
                except Exception:
                    cap_cols = None
                print(f"  [FUND] {market} columns: fundamental={fund_cols}, cap={cap_cols}")

                if df_fund_mkt is not None and df_cap_mkt is not None and not df_fund_mkt.empty and not df_cap_mkt.empty:
                    df_fund_mkt = df_fund_mkt.reset_index().rename(columns={"티커": "ticker"})
                    df_cap_mkt = df_cap_mkt.reset_index().rename(columns={"티커": "ticker", "종가": "close", "시가총액": "market_cap"})
                    required_fund_cols = ["ticker", "PER", "PBR", "EPS", "BPS"]
                    required_cap_cols = ["ticker", "close", "market_cap"]
                    missing_fund_cols = [col for col in required_fund_cols if col not in df_fund_mkt.columns]
                    missing_cap_cols = [col for col in required_cap_cols if col not in df_cap_mkt.columns]
                    if missing_fund_cols or missing_cap_cols:
                        print(
                            f"  [FUND][PYKRX] schema mismatch: market={market}, "
                            f"missing_fund={missing_fund_cols}, missing_cap={missing_cap_cols}"
                        )
                        continue
                    merged = pd.merge(df_fund_mkt, df_cap_mkt[["ticker", "close", "market_cap"]], on="ticker", how="inner")
                    for col in ["BPS", "PER", "PBR", "EPS", "DIV", "DPS", "close", "market_cap"]:
                        if col in merged.columns:
                            merged[col] = pd.to_numeric(merged[col], errors="coerce")

                    if not self._is_valid_fundamental_frame(merged):
                        print(
                            f"  [FUND][PYKRX] merged snapshot rejected: market={market}, rows={len(merged)}, reason=invalid_numeric_quality"
                        )
                        continue

                    merged["market"] = market
                    print(
                        f"  [FUND][PYKRX] merged: market={market}, rows={len(merged)}, cols={list(merged.columns)}"
                    )
                    self._set_fundamental_cache_lru(cache_key, merged)
                    dfs_merged.append(merged)
                    cache_misses += 1
                else:
                    print(
                        f"  [FUND][PYKRX] empty response: market={market}, "
                        f"fund_empty={df_fund_mkt is None or df_fund_mkt.empty}, "
                        f"cap_empty={df_cap_mkt is None or df_cap_mkt.empty}"
                    )
            except Exception:
                print(f"  [FUND] {market} fetch error: {traceback.format_exc()}")
                # 다음 마켓으로 계속 진행합니다
                continue

        self._log_timing(
            "fetch.fundamental_cap",
            time.perf_counter() - t_start,
            extra=f"hit={cache_hits}, miss={cache_misses}, markets={len(markets)}",
        )

        if len(dfs_merged) == 0:
            return pd.DataFrame()

        t_concat = time.perf_counter()
        try:
            shapes = [getattr(df, "shape", None) for df in dfs_merged]
            print(f"  [FUND] concat: parts={len(dfs_merged)}, shapes={shapes}")
            result_df = pd.concat(dfs_merged, ignore_index=True)
            self._log_timing(
                "fetch.fundamental_concat",
                time.perf_counter() - t_concat,
                extra=f"parts={len(dfs_merged)}",
            )
            return result_df
        except Exception:
            print("  [FUND] concat failed: returning empty dataframe")
            return pd.DataFrame()

    def _log_timing(self, label, elapsed_sec, extra=""):
        if not self.timing_enabled:
            return
        if extra:
            print(f"  [TIME] {label}: {elapsed_sec:.3f}s ({extra})")
        else:
            print(f"  [TIME] {label}: {elapsed_sec:.3f}s")

    def _log_filter_count(self, strategy, stage, before_count, after_count, extra=""):
        dropped = max(0, int(before_count) - int(after_count))
        if extra:
            print(
                f"  [{strategy}][COUNT] {stage}: {before_count} -> {after_count} "
                f"(drop={dropped}) ({extra})"
            )
        else:
            print(f"  [{strategy}][COUNT] {stage}: {before_count} -> {after_count} (drop={dropped})")

    def _log_numeric_column_stats(self, strategy, stage, df, columns):
        total = len(df)
        for col in columns:
            if col not in df.columns:
                print(f"  [{strategy}][STATS] {stage}.{col}: missing")
                continue

            series = pd.to_numeric(df[col], errors="coerce")
            valid = int(series.notna().sum())
            pos = int((series > 0).sum())

            if valid == 0:
                print(
                    f"  [{strategy}][STATS] {stage}.{col}: total={total}, valid=0, pos=0"
                )
                continue

            q = series.quantile([0.1, 0.5, 0.9])
            print(
                f"  [{strategy}][STATS] {stage}.{col}: total={total}, valid={valid}, pos={pos}, "
                f"min={series.min():.6g}, q10={q.loc[0.1]:.6g}, median={q.loc[0.5]:.6g}, "
                f"q90={q.loc[0.9]:.6g}, max={series.max():.6g}"
            )

    def get_market_tickers(self, date=None):
        try:
            if date is None:
                date = datetime.now().strftime("%Y%m%d")
            else:
                date = date.replace("-", "")
            normalized_date = self._normalize_date_yyyymmdd(date)

            if (not self._is_today(normalized_date)) and self._allow_kiwoom_date_proxy():
                print(
                    f"[TICKER] date proxy enabled: requested={normalized_date}, "
                    "kiwoom-current list will be used"
                )

            try:
                tickers = self._fetch_kiwoom_ticker_list_sync("KOSPI", normalized_date) + self._fetch_kiwoom_ticker_list_sync("KOSDAQ", normalized_date)
                tickers = list(dict.fromkeys(tickers))
                if len(tickers) > 0:
                    return tickers
            except Exception as e:
                print(f"Kiwoom 종목 리스트 조회 실패, pykrx fallback: {e}")

            if LIBRARIES_AVAILABLE:
                tickers_kospi = stock.get_market_ticker_list(date=normalized_date, market="KOSPI")
                tickers_kosdaq = stock.get_market_ticker_list(date=normalized_date, market="KOSDAQ")
                return list(set(tickers_kospi + tickers_kosdaq))
            return []
        except Exception as e:
            print(f"종목 리스트 조회 실패: {e}")
            return []

    def nearest_trading_date(self, date_str):
        """공휴일/주말이면 이후 가장 가까운 영업일 반환.
        get_nearest_business_day_in_a_week는 7일 제한이 있어
        추석 등 긴 연휴(10일+)에서 오동작한다.
        실제 거래 데이터(get_market_ohlcv_by_date)로 30일 범위를 탐색한다.
        """
        start_str = date_str.replace("-", "")
        dt = datetime.strptime(start_str, "%Y%m%d")
        end_str = (dt + timedelta(days=30)).strftime("%Y%m%d")
        try:
            df = stock.get_market_ohlcv_by_date(start_str, end_str, "005930")
            if not df.empty:
                return df.index[0].strftime("%Y%m%d")
        except Exception:
            pass
        # fallback: 최대 30일 탐색 (7일은 긴 연휴 대응 불가)
        for i in range(30):
            candidate = (dt + timedelta(days=i)).strftime("%Y%m%d")
            for market in ["KOSPI", "KOSDAQ"]:
                try:
                    df_test = stock.get_market_fundamental_by_ticker(candidate, market=market)
                    if not df_test.empty and df_test.iloc[:, 0].sum() != 0:
                        return candidate
                except Exception:
                    pass
        return start_str

    def previous_trading_date(self, date_str):
        """반드시 T-1 이전의 가장 가까운 영업일 반환.
        get_nearest_business_day_in_a_week는 7일 제한이 있어
        추석 등 긴 연휴(10일+)에서 오동작한다.
        실제 거래 데이터(get_market_ohlcv_by_date)로 30일 범위를 탐색한다.
        """
        dt = datetime.strptime(date_str.replace("-", ""), "%Y%m%d")
        prev_dt = dt - timedelta(days=1)
        end_str = prev_dt.strftime("%Y%m%d")
        start_str = (prev_dt - timedelta(days=30)).strftime("%Y%m%d")
        try:
            df = stock.get_market_ohlcv_by_date(start_str, end_str, "005930")
            if not df.empty:
                return df.index[-1].strftime("%Y%m%d")
        except Exception:
            pass
        # fallback: 최대 30일 탐색 (7일은 긴 연휴 대응 불가)
        for i in range(30):
            candidate = (prev_dt - timedelta(days=i)).strftime("%Y%m%d")
            for market in ["KOSPI", "KOSDAQ"]:
                try:
                    df_test = stock.get_market_fundamental_by_ticker(candidate, market=market)
                    if not df_test.empty and df_test.iloc[:, 0].sum() != 0:
                        return candidate
                except Exception:
                    pass
        return end_str

    def get_open_prices(self, tickers: list, date_str: str, fallback_prices: dict | None = None) -> tuple[dict, list]:
        """지정일(T)의 시가(Open)를 일괄 조회.

        pykrx ``get_market_ohlcv_by_ticker``로 시장별 일괄 요청하며,
        시가가 0이거나 조회 실패 시 fallback_prices(T-1 종가)로 대체한다.

        Args:
            tickers: 조회할 티커 목록.
            date_str: 시가 조회 기준일 (YYYY-MM-DD 또는 YYYYMMDD).
            fallback_prices: 시가 미조회 시 대체할 가격 맵 {ticker: price}.

        Returns:
            tuple[dict, list]: (시가 맵 {ticker: price}, fallback 대체된 ticker 목록).
        """
        if not LIBRARIES_AVAILABLE:
            fb = list(fallback_prices.keys()) if fallback_prices else []
            return (dict(fallback_prices) if fallback_prices else {}), fb

        date_yyyymmdd = date_str.replace('-', '')
        result: dict = {}

        found_tickers: set = set()
        zero_open_tickers: list = []
        failed_tickers: list = []

        for market in ["KOSPI", "KOSDAQ"]:
            try:
                df = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market=market)
                if df is None or df.empty or '시가' not in df.columns:
                    continue
                df.index = df.index.astype(str)
                for ticker in tickers:
                    if ticker in df.index and ticker not in result:
                        try:
                            open_price = float(df.loc[ticker, '시가'])
                            found_tickers.add(ticker)
                            if open_price > 0:
                                result[ticker] = open_price
                            else:
                                zero_open_tickers.append(ticker)
                        except Exception:
                            failed_tickers.append(ticker)
            except Exception as e:
                print(f"  [OPEN] {market} 시가 조회 실패: {type(e).__name__}: {e}")

        fallback_used: list = []
        missing_tickers: list = []
        if fallback_prices:
            for ticker in tickers:
                if ticker not in result:
                    if ticker in fallback_prices and fallback_prices[ticker]:
                        result[ticker] = float(fallback_prices[ticker])
                        fallback_used.append(ticker)
                    else:
                        missing_tickers.append(ticker)
        else:
            missing_tickers = [ticker for ticker in tickers if ticker not in result]

        if zero_open_tickers:
            print(
                f"  [OPEN] pykrx 반환값 중 시가=0: {len(zero_open_tickers)} "
                f"tickers={zero_open_tickers[:20]}{'' if len(zero_open_tickers) <= 20 else '...'}"
            )
        if failed_tickers:
            print(
                f"  [OPEN] pykrx에서 값 변환 실패: {len(failed_tickers)} "
                f"tickers={failed_tickers[:20]}{'' if len(failed_tickers) <= 20 else '...'}"
            )
        if fallback_used:
            print(
                f"  [OPEN] fallback 사용: {len(fallback_used)} "
                f"tickers={fallback_used[:20]}{'' if len(fallback_used) <= 20 else '...'}"
            )
        if missing_tickers:
            print(
                f"  [OPEN] 시가 미조회/대체 불가: {len(missing_tickers)} "
                f"tickers={missing_tickers[:20]}{'' if len(missing_tickers) <= 20 else '...'}"
            )

        n_open = sum(1 for t in tickers if t in result and (
            fallback_prices is None or result.get(t) != fallback_prices.get(t)
        ))
        print(f"  [OPEN] 시가 조회 완료: {n_open}/{len(tickers)} pykrx, "
              f"fallback={len(tickers) - n_open}")
        return result, fallback_used

    def get_close_prices(self, tickers: list, date_str: str, fallback_prices: dict | None = None) -> tuple[dict, list]:
        """지정일(T)의 종가(Close)를 일괄 조회.

        pykrx ``get_market_ohlcv_by_ticker``로 시장별 일괄 요청하며,
        종가가 0이거나 조회 실패 시 fallback_prices(T-1 종가)로 대체한다.
        장마감 동시호가(15:20) 체결 방식의 백테스트에서 실행가로 사용된다.

        Args:
            tickers: 조회할 티커 목록.
            date_str: 종가 조회 기준일 (YYYY-MM-DD 또는 YYYYMMDD).
            fallback_prices: 종가 미조회 시 대체할 가격 맵 {ticker: price}.

        Returns:
            tuple[dict, list]: (종가 맵 {ticker: price}, fallback 대체된 ticker 목록).
        """
        if not LIBRARIES_AVAILABLE:
            fb = list(fallback_prices.keys()) if fallback_prices else []
            return (dict(fallback_prices) if fallback_prices else {}), fb

        date_yyyymmdd = date_str.replace('-', '')
        result: dict = {}

        found_tickers: set = set()
        zero_close_tickers: list = []
        failed_tickers: list = []

        for market in ["KOSPI", "KOSDAQ"]:
            try:
                df = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market=market)
                if df is None or df.empty or '종가' not in df.columns:
                    continue
                df.index = df.index.astype(str)
                for ticker in tickers:
                    if ticker in df.index and ticker not in result:
                        try:
                            close_price = float(df.loc[ticker, '종가'])
                            found_tickers.add(ticker)
                            if close_price > 0:
                                result[ticker] = close_price
                            else:
                                zero_close_tickers.append(ticker)
                        except Exception:
                            failed_tickers.append(ticker)
            except Exception as e:
                print(f"  [CLOSE] {market} 종가 조회 실패: {type(e).__name__}: {e}")

        fallback_used: list = []
        missing_tickers: list = []
        if fallback_prices:
            for ticker in tickers:
                if ticker not in result:
                    if ticker in fallback_prices and fallback_prices[ticker]:
                        result[ticker] = float(fallback_prices[ticker])
                        fallback_used.append(ticker)
                    else:
                        missing_tickers.append(ticker)
        else:
            missing_tickers = [ticker for ticker in tickers if ticker not in result]

        if zero_close_tickers:
            print(
                f"  [CLOSE] pykrx 반환값 중 종가=0: {len(zero_close_tickers)} "
                f"tickers={zero_close_tickers[:20]}{'' if len(zero_close_tickers) <= 20 else '...'}"
            )
        if failed_tickers:
            print(
                f"  [CLOSE] pykrx에서 값 변환 실패: {len(failed_tickers)} "
                f"tickers={failed_tickers[:20]}{'' if len(failed_tickers) <= 20 else '...'}"
            )
        if fallback_used:
            print(
                f"  [CLOSE] fallback 사용: {len(fallback_used)} "
                f"tickers={fallback_used[:20]}{'' if len(fallback_used) <= 20 else '...'}"
            )
        if missing_tickers:
            print(
                f"  [CLOSE] 종가 미조회/대체 불가: {len(missing_tickers)} "
                f"tickers={missing_tickers[:20]}{'' if len(missing_tickers) <= 20 else '...'}"
            )

        n_close = sum(1 for t in tickers if t in result and (
            fallback_prices is None or result.get(t) != fallback_prices.get(t)
        ))
        print(f"  [CLOSE] 종가 조회 완료: {n_close}/{len(tickers)} pykrx, "
              f"fallback={len(tickers) - n_close}")
        return result, fallback_used

    def _get_industry_info(self, tickers, date_str):
        try:
            industry_map = {}
            # 캐시 미스 종목만 추려냄
            missing = [t for t in set(tickers) if t not in self.industry_cache]
            cache_hits = len(set(tickers)) - len(missing)

            if missing:
                # 배치 조회: 업종분류현황().fetch() 직접 호출 (per-ticker API 함수 없음)
                for mkt_code in ["STK", "KSQ"]:
                    try:
                        from pykrx.website.krx.market.wrap import 업종분류현황
                        raw = 업종분류현황().fetch(date_str, mkt_code)
                        if raw is not None and not raw.empty:
                            code_col = next((c for c in ["ISU_SRT_CD"] if c in raw.columns), None)
                            name_col = next((c for c in ["IDX_IND_NM"] if c in raw.columns), None)
                            if code_col and name_col:
                                for _, row in raw.iterrows():
                                    ticker = str(row[code_col]).strip().zfill(6)
                                    sector = str(row.get(name_col, "기타")).strip() or "기타"
                                    self.industry_cache[ticker] = sector
                    except Exception:
                        pass

                # 배치 조회 후에도 미스인 종목은 "기타"로 채움
                for ticker in missing:
                    if ticker not in self.industry_cache:
                        self.industry_cache[ticker] = "기타"

            for ticker in tickers:
                industry_map[ticker] = self.industry_cache.get(ticker, "기타")

            cache_miss = len(missing)
            if self.timing_enabled:
                print(f"    [CACHE] industry hit={cache_hits}, miss={cache_miss}")
            return industry_map
        except Exception:
            return {t: "기타" for t in tickers}

    def _pick_dividend_column(self, df):
        if "DIV" in df.columns:
            return "DIV"
        if "배당수익률" in df.columns:
            return "배당수익률"
        return None

    def screen_stocks_pykrx_roe(self, target_date, markets=["KOSPI", "KOSDAQ"]):
        if not LIBRARIES_AVAILABLE:
            return pd.DataFrame()

        try:
            date_str = target_date.replace("-", "")
            df = self._get_fundamental_and_cap(date_str, markets)

            if df.empty:
                print("    ROE 스크리닝: 펀더멘탈 데이터 없음")
                return pd.DataFrame()

            industry_map = self._get_industry_info(df["ticker"].tolist(), date_str)
            df["industry"] = df["ticker"].map(industry_map).fillna("기타")

            df = df[(df["PER"] > 0) & (df["PER"] < 20)]
            df = df[(df["PBR"] > 0.2) & (df["PBR"] < 10)]
            df = df[df["market_cap"] >= 5e10]

            if len(df) == 0:
                print("    ROE 스크리닝: 필터링 후 종목 없음")
                return pd.DataFrame()

            df.loc[:, "ROE"] = np.where(
                (df["EPS"] > 0) & (df["BPS"] > 0),
                (df["EPS"] / df["BPS"]) * 100,
                np.nan,
            )
            df = df[df["ROE"] >= 10]

            t_roe_rank = time.perf_counter()
            try:
                df.loc[:, "rank_per"] = df["PER"].rank(ascending=True, method="average", na_option="bottom")
                df.loc[:, "rank_pbr"] = df["PBR"].rank(ascending=True, method="average", na_option="bottom")
                df.loc[:, "rank_roe"] = df["ROE"].rank(ascending=False, method="average", na_option="bottom")
                df.loc[:, "total_rank"] = df["rank_per"] + df["rank_pbr"] + df["rank_roe"]
            finally:
                self._log_timing(
                    "screen.roe.ranking",
                    time.perf_counter() - t_roe_rank,
                    extra=f"rows={len(df)}",
                )

            result = df.sort_values("total_rank", ascending=True).head(self.num_stocks)

            if len(result) > 0:
                print(f"      필터 후: {len(df)}개 → 선정: {len(result)}개")
                print(f"      PER {result['PER'].min():.1f}~{result['PER'].max():.1f} (평균 {result['PER'].mean():.2f})")
                print(f"      PBR {result['PBR'].min():.2f}~{result['PBR'].max():.2f} (평균 {result['PBR'].mean():.2f})")
                print(f"      ROE {result['ROE'].min():.1f}~{result['ROE'].max():.1f}% (평균 {result['ROE'].mean():.2f}%)")

            return result[["ticker", "PER", "PBR", "ROE", "total_rank", "close", "market_cap"]].copy()

        except Exception as e:
            print(f"    ROE 스크리닝 오류: {e}")
            return pd.DataFrame()

    def screen_stocks_mixed(self, target_date, markets=["KOSPI", "KOSDAQ"]):
        if not LIBRARIES_AVAILABLE:
            return pd.DataFrame()

        try:
            t_screen_start = time.perf_counter()
            date_str = target_date.replace("-", "")
            df = self._get_fundamental_and_cap(date_str, markets)

            if df.empty:
                print("    MIXED 스크리닝: 펀더멘탈 데이터 없음")
                return pd.DataFrame()

            self._log_filter_count("MIXED", "input", len(df), len(df), extra=f"profile={self.mixed_filter_profile}")
            self._log_numeric_column_stats(
                "MIXED",
                "pre_base",
                df,
                ["PER", "PBR", "market_cap"],
            )

            before = len(df)
            if self.large_cap_min_mcap is None:
                df = df[(df["PER"] > 0) & (df["PBR"] > 0)]
            else:
                try:
                    min_mcap = float(self.large_cap_min_mcap)
                except Exception:
                    min_mcap = None

                if min_mcap is None:
                    df = df[(df["PER"] > 0) & (df["PBR"] > 0)]
                else:
                    df = df[(df["PER"] > 0) & (df["PBR"] > 0) & (df["market_cap"] >= min_mcap)]
            self._log_filter_count("MIXED", "base_filter", before, len(df))

            df.loc[:, "ROE"] = np.where(
                (df["EPS"] > 0) & (df["BPS"] > 0),
                (df["EPS"] / df["BPS"]) * 100,
                np.nan,
            )
            before = len(df)
            df = df[df["ROE"] >= 10]
            self._log_filter_count("MIXED", "quality_roe>=10", before, len(df))

            dividend_col = self._pick_dividend_column(df)
            if dividend_col is None:
                df.loc[:, "DIV_YIELD"] = 0.0
            else:
                df.loc[:, "DIV_YIELD"] = pd.to_numeric(df[dividend_col], errors="coerce").fillna(0.0)

            if len(df) == 0:
                print("    MIXED 스크리닝: 품질 필터 후 종목 없음")
                return pd.DataFrame()

            before = len(df)
            cap_lower_limit = df["market_cap"].quantile(0.80)
            df = df[df["market_cap"] >= cap_lower_limit]
            if self.large_cap_min_mcap is not None:
                try:
                    min_mcap = float(self.large_cap_min_mcap)
                    df = df[df["market_cap"] >= min_mcap]
                except Exception:
                    pass
            self._log_filter_count("MIXED", "profile.large_cap", before, len(df), extra=f"q80={cap_lower_limit:,.0f}")

            if len(df) == 0:
                print("    MIXED 스크리닝: 시장별 분위수 필터 후 종목 없음")
                return pd.DataFrame()

            t_ranking_start = time.perf_counter()
            try:
                df.loc[:, "rank_per"] = df.groupby("market")["PER"].rank(ascending=True, method="average", na_option="bottom")
                df.loc[:, "rank_pbr"] = df.groupby("market")["PBR"].rank(ascending=True, method="average", na_option="bottom")
                df.loc[:, "rank_roe"] = df.groupby("market")["ROE"].rank(ascending=False, method="average", na_option="bottom")
                df.loc[:, "rank_div"] = df.groupby("market")["DIV_YIELD"].rank(ascending=False, method="average", na_option="bottom")

                value_rank = df["rank_per"] + df["rank_pbr"]
            finally:
                self._log_timing(
                    "screen.mixed.ranking",
                    time.perf_counter() - t_ranking_start,
                    extra=f"profile={self.mixed_filter_profile}, rows={len(df)}",
                )

            if self.momentum_enabled:
                t_mom_start = time.perf_counter()
                moms = []
                cache_hit = 0
                cache_miss = 0
                try:
                    end_dt = pd.to_datetime(date_str)
                    start_dt = (end_dt - pd.DateOffset(months=self.momentum_months)).strftime("%Y%m%d")
                    end_dt_str = end_dt.strftime("%Y%m%d")
                except Exception:
                    start_dt = date_str
                    end_dt_str = date_str

                tickers_list = list(df["ticker"])
                total_universe = len(tickers_list)

                to_fetch: list[tuple[str, str]] = []
                for ticker in tickers_list:
                    cache_key = f"{ticker}|{start_dt}|{end_dt_str}"
                    if cache_key in self.momentum_cache:
                        moms.append(self.momentum_cache[cache_key])
                        cache_hit += 1
                    else:
                        to_fetch.append((ticker, cache_key))

                if len(to_fetch) > 0:
                    concurrency = max(1, int(env_get("MOMENTUM_CONCURRENCY", default="8")))
                    per_call_timeout = float(env_get("MOMENTUM_TIMEOUT", default="8.0"))

                    def _fetch_one(ticker: str, cache_key: str):
                            # 시작/종료에 대해 price_cache를 확인하여 전체 OHLCV 재요청을 피합니다
                        start_key = f"{start_dt}|{ticker}"
                        end_key = f"{end_dt_str}|{ticker}"
                        try:
                            if start_key in self.price_cache and end_key in self.price_cache:
                                first = float(self.price_cache[start_key])
                                last = float(self.price_cache[end_key])
                                mom = (last / first) - 1 if first > 0 else 0.0
                                return (ticker, cache_key, float(mom), 0.0, True)
                        except Exception:
                            pass

                        retries = max(0, int(env_get("MOMENTUM_RETRIES", default="2")))
                        backoff_base = float(env_get("MOMENTUM_BACKOFF", default="0.5"))
                        t0 = time.perf_counter()
                        for attempt in range(retries + 1):
                            try:
                                ohlc = stock.get_market_ohlcv(start_dt, end_dt_str, ticker)
                                dur = time.perf_counter() - t0
                                if ohlc is not None and not ohlc.empty:
                                    first = ohlc["종가"].iloc[0]
                                    last = ohlc["종가"].iloc[-1]
                                    mom = (last / first) - 1 if first > 0 else 0.0
                                    # 시작/종료 시점에 대한 price cache를 채웁니다
                                    try:
                                        self.price_cache[start_key] = float(first)
                                        self.price_cache[end_key] = float(last)
                                    except Exception:
                                        pass
                                else:
                                    mom = 0.0
                                return (ticker, cache_key, float(mom), dur, True)
                            except Exception:
                                dur = time.perf_counter() - t0
                                if attempt < retries:
                                    sleep_for = backoff_base * (2 ** attempt)
                                    time.sleep(sleep_for)
                                    continue
                                return (ticker, cache_key, 0.0, dur, False)

                    completed = 0
                    ex = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
                    future_to_ticker = {ex.submit(_fetch_one, t, ck): (t, ck) for t, ck in to_fetch}
                    try:
                        for fut in concurrent.futures.as_completed(future_to_ticker):
                            try:
                                ticker, cache_key, mom, dur, success = fut.result(timeout=per_call_timeout)
                            except concurrent.futures.TimeoutError:
                                tinfo = future_to_ticker.get(fut)
                                ticker, cache_key, mom, dur, success = (tinfo[0], tinfo[1], 0.0, per_call_timeout, False)
                            except Exception:
                                tinfo = future_to_ticker.get(fut)
                                ticker, cache_key, mom, dur, success = (tinfo[0], tinfo[1], 0.0, 0.0, False)

                            self.momentum_cache[cache_key] = float(mom)
                            cache_miss += 1
                            completed += 1

                            # 진행상황을 로깅하고 느린 호출을 기록합니다
                            if (cache_hit + cache_miss) % 20 == 0 or completed % 20 == 0:
                                print(f"  [FUND][MOM] progress: {cache_hit+cache_miss}/{total_universe}, last={ticker}, dur={dur:.3f}s, hits={cache_hit}, miss={cache_miss}")
                            if dur > 0.5:
                                print(f"  [FUND][MOM] slow_ohlcv: ticker={ticker}, dur={dur:.3f}s")
                    except KeyboardInterrupt:
                        print("  [FUND][MOM] interrupted: cancelling pending momentum fetches")
                        for fut in future_to_ticker:
                            try:
                                fut.cancel()
                            except Exception:
                                pass
                        try:
                            ex.shutdown(wait=False)
                        except Exception:
                            pass
                        raise
                    finally:
                        try:
                            ex.shutdown(wait=False)
                        except Exception:
                            pass

                    # 원본 순서대로 모멘텀 값을 채웁니다
                    moms = [self.momentum_cache.get(f"{t}|{start_dt}|{end_dt_str}", 0.0) for t in tickers_list]

                self._log_timing(
                    "mixed.momentum",
                    time.perf_counter() - t_mom_start,
                    extra=f"hit={cache_hit}, miss={cache_miss}, universe={len(df)}",
                )

                df.loc[:, "mom"] = moms
                if self.momentum_filter_enabled:
                    before = len(df)
                    df = df[df["mom"] > 0]
                    self._log_filter_count("MIXED", "momentum>0", before, len(df), extra="momentum_filter_enabled=true")
                df.loc[:, "rank_mom_pct"] = df.groupby("market")["mom"].rank(ascending=False, pct=True, method="average")
            else:
                df["rank_mom_pct"] = 0.0

            # 2단계 혼합 방식
            # Stage 1: 순수 그린블라트 등수 합산 (단위 무관, 임의 가중 없음)
            df.loc[:, "greenblatt_rank"] = value_rank + df["rank_roe"]

            # Stage 2: 그린블라트 합산을 [0,1] 정규화 후 momentum_weight 비율로 모멘텀과 혼합
            df.loc[:, "rank_greenblatt_pct"] = df.groupby("market")["greenblatt_rank"].rank(ascending=True, pct=True, method="average")
            m = float(self.momentum_weight) if self.momentum_enabled else 0.0
            df.loc[:, "total_rank"] = (1.0 - m) * df["rank_greenblatt_pct"] + m * df["rank_mom_pct"]

            df_sorted = df.sort_values("total_rank", ascending=True)

            # 섹터 분산 제한: 동일 업종 최대 sector_max_stocks 종목까지만 허용
            if self.sector_max_stocks and self.sector_max_stocks > 0:
                industry_map = self._get_industry_info(df_sorted["ticker"].tolist(), date_str)
                df_sorted = df_sorted.copy()
                df_sorted["industry"] = df_sorted["ticker"].map(industry_map).fillna("기타")
                sector_counts: dict[str, int] = {}
                picked_indices = []
                for idx, row in df_sorted.iterrows():
                    sec = row["industry"]
                    cnt = sector_counts.get(sec, 0)
                    if cnt < self.sector_max_stocks:
                        picked_indices.append(idx)
                        sector_counts[sec] = cnt + 1
                    if len(picked_indices) >= self.num_stocks:
                        break
                df_sorted = df_sorted.loc[picked_indices]
                print(f"      [SECTOR] max_per_sector={self.sector_max_stocks}, 선정: {len(df_sorted)}개")

            if self.kosdaq_target_ratio is not None:
                kosdaq_target = int(self.num_stocks * self.kosdaq_target_ratio)
                kospi_target = self.num_stocks - kosdaq_target

                kosdaq_pool = df_sorted[df_sorted["market"] == "KOSDAQ"]
                kospi_pool = df_sorted[df_sorted["market"] == "KOSPI"]

                selected = []
                if kosdaq_target > 0:
                    selected.append(kosdaq_pool.head(kosdaq_target))
                if kospi_target > 0:
                    selected.append(kospi_pool.head(kospi_target))

                if len(selected) > 0:
                    result = pd.concat(selected, ignore_index=True)
                else:
                    result = df_sorted.head(self.num_stocks)

                if len(result) < self.num_stocks:
                    remaining = self.num_stocks - len(result)
                    chosen = set(result["ticker"])
                    extra = df_sorted[~df_sorted["ticker"].isin(chosen)].head(remaining)
                    if not extra.empty:
                        result = pd.concat([result, extra], ignore_index=True)

                result = result.head(self.num_stocks)
            else:
                result = df_sorted.head(self.num_stocks)

            if len(result) > 0:
                market_counts = result["market"].value_counts().to_dict()
                print(f"      [MIXED-{self.mixed_filter_profile}] 필터 후: {len(df)}개 → 선정: {len(result)}개")
                self._log_filter_count("MIXED", "final_selection", len(df), len(result), extra=f"num_stocks={self.num_stocks}")
                print(f"      [MIXED] 시장구성: {market_counts}")
                print(f"      [MIXED] PER 평균 {result['PER'].mean():.2f}, PBR 평균 {result['PBR'].mean():.2f}")
                print(f"      [MIXED] ROE 평균 {result['ROE'].mean():.2f}%, 배당수익률 평균 {result['DIV_YIELD'].mean():.2f}%")

            self._log_timing("mixed.total", time.perf_counter() - t_screen_start)

            return result[["ticker", "market", "PER", "PBR", "ROE", "DIV_YIELD", "total_rank", "close", "market_cap"]].copy()

        except Exception as e:
            print(f"    MIXED 스크리닝 오류: {e}")
            return pd.DataFrame()

    def select_stocks(self, trading_date: str) -> pd.DataFrame:
        if self.strategy_mode == "mixed":
            return self.screen_stocks_mixed(trading_date)
        return self.screen_stocks_pykrx_roe(trading_date)
