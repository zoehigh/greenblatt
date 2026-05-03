from __future__ import annotations

import argparse
import asyncio
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time
import fcntl
import json
import os
from typing import Any, Literal

import pandas as pd

from live_trading.config import LiveTradingConfig
from live_trading.execution import build_order_intents, select_capital_constrained_stocks
from live_trading.kiwoom_adapter import KiwoomBrokerAdapter, KiwoomAPIError
from live_trading.kiwoom_http_patch import apply_kiwoom_client_session_patch
from live_trading.strategy_bridge import build_rebalance_signal
from live_trading.strategy_engine import LiveSignalEngine
from vol_targeting import compute_vol_target_ratio, warmup_vol_history


ExecutionState = Literal[
    "STARTED",
    "FAILED_BEFORE_ORDER",
    "ORDER_SUBMITTED",
    "PARTIAL_PENDING",
    "SUCCESS",
    "SKIPPED",
]

ExecutionAction = Literal["full_rebalance", "reconcile_only", "skip"]


def _map_legacy_result_to_execution_state(last_result: str) -> ExecutionState | None:
    normalized = (last_result or "").strip().lower()
    mapping: dict[str, ExecutionState] = {
        "executed": "SUCCESS",
        "empty_selection": "SKIPPED",
        "no_intent": "SKIPPED",
    }
    return mapping.get(normalized)


def _resolve_execution_state_from_state(state: dict[str, Any]) -> ExecutionState | None:
    raw = state.get("execution_state")
    if isinstance(raw, str):
        normalized = raw.strip().upper()
        if normalized in {
            "STARTED",
            "FAILED_BEFORE_ORDER",
            "ORDER_SUBMITTED",
            "PARTIAL_PENDING",
            "SUCCESS",
            "SKIPPED",
        }:
            return normalized  # type: ignore[return-value]

    legacy = state.get("last_result")
    if isinstance(legacy, str):
        return _map_legacy_result_to_execution_state(legacy)

    return None


def _kst_now() -> datetime:
    return datetime.now(tz=ZoneInfo("Asia/Seoul"))


def _kst_date_string() -> str:
    return _kst_now().strftime("%Y-%m-%d")


def _limit_price(side: str, ref_price: float, bps: int) -> int:
    if ref_price <= 0:
        return 0
    factor = 1 + (bps / 10000) if side == "BUY" else 1 - (bps / 10000)
    return max(1, int(ref_price * factor))


async def _wait_until_order_time(config: LiveTradingConfig) -> None:
    if not config.order_time_wait_enabled:
        return

    hhmm = str(config.order_time_hhmm or "").strip()
    try:
        hh, mm = hhmm.split(":", 1)
        now = _kst_now()
        target = datetime(
            year=now.year,
            month=now.month,
            day=now.day,
            hour=int(hh),
            minute=int(mm),
            second=0,
            microsecond=0,
            tzinfo=ZoneInfo("Asia/Seoul"),
        )
    except Exception as exc:
        raise ValueError(
            f"Invalid order_time_hhmm value: {config.order_time_hhmm!r}. "
            "Expected format HH:MM."
        ) from exc

    if now >= target:
        return

    delta_seconds = (target - now).total_seconds()
    wait_seconds = math.ceil(delta_seconds) + max(0, config.order_time_grace_seconds)
    if wait_seconds > 0:
        print(f"주문 시각 대기: {wait_seconds}초 ({config.order_time_hhmm} 목표)")
        await asyncio.sleep(wait_seconds)


async def _sweep_cancel_open_orders(broker: KiwoomBrokerAdapter) -> int:
    """미체결 주문 전량을 취소하고 취소한 건수를 반환한다.

    full_rebalance 진입 시 이전 주기의 잔여 미체결이 중복 주문되는 것을 방지한다.
    취소 실패한 건은 경고 로그만 남기고 진행한다.
    """
    try:
        unfilled = await broker.query_unfilled_orders(stex_tp="0")
    except Exception as exc:
        print(f"[SWEEP] 미체결 조회 실패 (취소 생략): {exc}")
        return 0

    if not unfilled:
        print("[SWEEP] 미체결 주문 없음, 취소 생략")
        return 0

    print(f"[SWEEP] 미체결 주문 {len(unfilled)}건 취소 시작")
    cancelled = 0
    for rec in unfilled:
        ord_no_raw = rec.get("ord_no") or rec.get("주문번호") or rec.get("9203") or ""
        stk_cd_raw = rec.get("stk_cd") or rec.get("종목코드") or ""
        try:
            rem = int(float(rec.get("ord_remnq") or rec.get("미체결수량") or 0))
        except Exception:
            rem = 0

        if not ord_no_raw or rem <= 0:
            continue

        from live_trading.kiwoom_adapter import SubmittedOrder
        dummy = SubmittedOrder(
            ticker=str(stk_cd_raw).lstrip("A"),
            side="UNKNOWN",
            requested_qty=rem,
            price=0,
            order_no=str(ord_no_raw),
            raw_response={},
        )
        try:
            await broker.cancel_order(dummy, rem)
            print(f"[SWEEP] 취소 완료: ord_no={ord_no_raw}, ticker={stk_cd_raw}, qty={rem}")
            cancelled += 1
        except Exception as exc:
            print(f"[SWEEP] 취소 실패 (무시): ord_no={ord_no_raw}, ticker={stk_cd_raw}, err={exc}")

    print(f"[SWEEP] 취소 완료: {cancelled}/{len(unfilled)}건")
    return cancelled


async def _reconcile_pending_orders(
    broker: KiwoomBrokerAdapter,
    report_rows: list[dict],
    trading_date: str,
) -> bool:
    """이전 주기의 미체결 주문을 처리한다.

    1) 미체결 주문 목록 조회
    2) 이미 체결된 주문은 fills 기록
    3) 미체결 잔량은 정정주문(modify) 우선 시도 → 실패 시 취소+재주문 폴백
       - 정정주문: 단일 API 호출로 원자적 처리, 취소-재주문 사이 가격 공백 없음
       - 폴백: 정정 API 자체가 실패(Rate Limit 등)한 경우에만 사용
    완전히 처리됐으면 True, 미체결이 남아 있으면 False 반환.
    """
    try:
        unfilled = await broker.query_unfilled_orders(stex_tp="0")
    except Exception as exc:
        print(f"[RECONCILE] 미체결 조회 실패: {exc}")
        return False

    if not unfilled:
        print("[RECONCILE] 미체결 주문 없음, 완료로 간주")
        return True

    print(f"[RECONCILE] 미체결 주문 {len(unfilled)}건 처리 시작")
    ts = _kst_now().strftime("%Y-%m-%d %H:%M:%S")
    remaining = 0

    from live_trading.kiwoom_adapter import SubmittedOrder
    for rec in unfilled:
        ord_no_raw = rec.get("ord_no") or rec.get("주문번호") or ""
        stk_cd_raw = rec.get("stk_cd") or rec.get("종목코드") or ""
        ticker = str(stk_cd_raw).lstrip("A")
        try:
            rem_qty = int(float(rec.get("ord_remnq") or rec.get("미체결수량") or 0))
        except Exception:
            rem_qty = 0
        try:
            ord_qty = int(float(rec.get("ord_qty") or rec.get("주문수량") or 0))
        except Exception:
            ord_qty = 0
        try:
            ord_price = int(float(rec.get("ord_uv") or rec.get("주문단가") or 0))
        except Exception:
            ord_price = 0
        side_raw = str(rec.get("buy_sell_tp") or rec.get("매매구분") or "").strip()
        side = "BUY" if side_raw in {"1", "매수"} else "SELL" if side_raw in {"2", "매도"} else "UNKNOWN"

        filled_qty = max(0, ord_qty - rem_qty)

        # 체결된 수량이 있으면 fills 기록
        if filled_qty > 0:
            report_rows.append({
                "timestamp": ts,
                "round": "reconcile_fill",
                "ticker": ticker,
                "side": side,
                "order_no": str(ord_no_raw),
                "order_price": ord_price,
                "requested_qty": ord_qty,
                "pending_qty": rem_qty,
                "is_filled": rem_qty == 0,
                "return_code": None,
                "note": "reconcile",
            })

        if rem_qty <= 0:
            continue

        if side == "UNKNOWN":
            print(f"[RECONCILE] side 불명확, 처리 생략: ord_no={ord_no_raw}, ticker={ticker}")
            remaining += 1
            continue

        dummy = SubmittedOrder(
            ticker=ticker,
            side=side,
            requested_qty=rem_qty,
            price=ord_price,
            order_no=str(ord_no_raw),
            raw_response={},
        )

        retry_price = await broker.get_retry_price(side=side, ticker=ticker, base_price=ord_price)

        # 1단계: 정정주문 시도 (원자적, API 호출 1회)
        modify_succeeded = False
        try:
            mod_resp = await broker.modify_order(
                order=dummy,
                new_qty=rem_qty,
                new_price=max(1, retry_price),
            )
            new_ord_no = str(mod_resp.get("ord_no") or mod_resp.get("base_orig_ord_no") or ord_no_raw)
            print(f"[RECONCILE] 정정 완료: ord_no={ord_no_raw}→{new_ord_no}, ticker={ticker}, qty={rem_qty}, price={retry_price}")
            report_rows.append({
                "timestamp": _kst_now().strftime("%Y-%m-%d %H:%M:%S"),
                "round": "reconcile_modify",
                "ticker": ticker,
                "side": side,
                "order_no": new_ord_no,
                "order_price": max(1, retry_price),
                "requested_qty": rem_qty,
                "pending_qty": rem_qty,
                "is_filled": False,
                "return_code": mod_resp.get("return_code"),
                "note": "reconcile_modify",
            })
            modify_succeeded = True
        except Exception as mod_exc:
            print(f"[RECONCILE] 정정 실패, 취소+재주문 폴백: ord_no={ord_no_raw}, ticker={ticker}, err={mod_exc}")

        if modify_succeeded:
            continue

        # 2단계: 폴백 — 취소 후 재주문
        try:
            await broker.cancel_order(dummy, rem_qty)
            print(f"[RECONCILE] 취소: ord_no={ord_no_raw}, ticker={ticker}, qty={rem_qty}")
        except Exception as cancel_exc:
            print(f"[RECONCILE] 취소 실패: ord_no={ord_no_raw}, ticker={ticker}, err={cancel_exc}")
            remaining += 1
            continue

        try:
            retried = await broker.submit_order(
                ticker=ticker,
                side=side,
                quantity=rem_qty,
                price=max(1, retry_price),
            )
            print(f"[RECONCILE] 재주문: ticker={ticker}, side={side}, qty={rem_qty}, price={retry_price}, new_ord_no={retried.order_no}")
            report_rows.append({
                "timestamp": _kst_now().strftime("%Y-%m-%d %H:%M:%S"),
                "round": "reconcile_resubmit",
                "ticker": ticker,
                "side": side,
                "order_no": retried.order_no,
                "order_price": retried.price,
                "requested_qty": rem_qty,
                "pending_qty": rem_qty,
                "is_filled": False,
                "return_code": retried.raw_response.get("return_code"),
                "note": "reconcile_resubmit",
            })
        except Exception as submit_exc:
            print(f"[RECONCILE] 재주문 실패: ticker={ticker}, side={side}, err={submit_exc}")
            remaining += 1

    print(f"[RECONCILE] 처리 완료. 미해결 잔여: {remaining}건")
    return remaining == 0


def _append_report_rows(report_rows: list[dict], checks: list, round_name: str) -> None:
    ts = _kst_now().strftime("%Y-%m-%d %H:%M:%S")
    for check in checks:
        report_rows.append(
            {
                "timestamp": ts,
                "round": round_name,
                "ticker": check.submitted.ticker,
                "side": check.submitted.side,
                "order_no": check.submitted.order_no,
                "order_price": check.submitted.price,
                "requested_qty": check.submitted.requested_qty,
                "pending_qty": check.pending_qty,
                "is_filled": check.is_filled,
                "return_code": check.submitted.raw_response.get("return_code"),
            }
        )


def _print_selected_debug(selected: pd.DataFrame, max_rows: int) -> None:
    columns = [
        "ticker",
        "name",
        "market",
        "close",
        "per",
        "pbr",
        "roe",
        "score",
    ]
    available_cols = [col for col in columns if col in selected.columns]
    if not available_cols:
        available_cols = list(selected.columns)

    preview = selected[available_cols].copy()
    for col in preview.columns:
        if pd.api.types.is_numeric_dtype(preview[col]):
            preview[col] = preview[col].round(4)

    print(f"[DEBUG] 선정 종목 미리보기: total={len(selected)}")
    print(preview.head(max_rows).to_string(index=False))
    if len(preview) > max_rows:
        print(f"[DEBUG] 선정 종목 출력 생략: {len(preview) - max_rows}개")


def _print_intents_debug(intents: list, max_rows: int, order_price_offset_bps: int) -> None:
    if not intents:
        print("[DEBUG] 주문 의도 없음")
        return

    rows = []
    for intent in intents:
        rows.append(
            {
                "ticker": intent.ticker,
                "side": intent.side,
                "qty": intent.quantity,
                "current_qty": intent.current_quantity,
                "target_qty": intent.target_quantity,
                "ref_price": round(float(intent.reference_price), 2),
                "limit_price": _limit_price(intent.side, intent.reference_price, order_price_offset_bps),
                "reason": intent.reason,
            }
        )

    frame = pd.DataFrame(rows)
    buy_count = int((frame["side"] == "BUY").sum())
    sell_count = int((frame["side"] == "SELL").sum())
    print(f"[DEBUG] 주문 의도 미리보기: total={len(frame)} (BUY={buy_count}, SELL={sell_count})")
    print(frame.head(max_rows).to_string(index=False))
    if len(frame) > max_rows:
        print(f"[DEBUG] 주문 의도 출력 생략: {len(frame) - max_rows}개")


def _save_daily_report(config: LiveTradingConfig, trading_date: str, report_rows: list[dict]) -> None:
    if not config.save_daily_report:
        return
    if not report_rows:
        return

    os.makedirs(config.report_dir, exist_ok=True)
    file_name = f"fills_{trading_date.replace('-', '')}.csv"
    path = os.path.join(config.report_dir, file_name)
    pd.DataFrame(report_rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"일일 체결 리포트 저장: {path}")


def _save_rebalance_snapshot(
    config: LiveTradingConfig,
    trading_date: str,
    selected: pd.DataFrame,
    holdings: dict[str, int],
    cash: float,
    effective_investment_ratio: float,
    vol_multiplier: float | None = None,
    mt_in_market: bool | None = None,
    mt_reason: str | None = None,
) -> None:
    """리밸런싱 시점의 포트폴리오 스냅샷, 선정 종목, 타이밍 결정을 저장한다.

    results/live_reports/snapshot_{yyyymmdd}.json 으로 저장.
    이후 성과 분석 및 실전 vs 백테스트 비교에 활용한다.
    """
    if not config.save_daily_report:
        return

    os.makedirs(config.report_dir, exist_ok=True)
    yyyymmdd = trading_date.replace("-", "")
    snap_path = os.path.join(config.report_dir, f"snapshot_{yyyymmdd}.json")

    snapshot_data: dict[str, Any] = {
        "trading_date": trading_date,
        "timestamp": _kst_now().strftime("%Y-%m-%d %H:%M:%S"),
        "cash": cash,
        "effective_investment_ratio": effective_investment_ratio,
        "vol_target_multiplier": vol_multiplier,
        "market_timing_in_market": mt_in_market,
        "market_timing_reason": mt_reason,
        "holdings": [
            {"ticker": t, "quantity": q} for t, q in sorted(holdings.items())
        ],
        "selected_stocks": (
            selected[["ticker", "name", "close", "score"]].to_dict(orient="records")
            if not selected.empty and all(c in selected.columns for c in ["ticker", "name", "close", "score"])
            else selected.to_dict(orient="records") if not selected.empty
            else []
        ),
    }

    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(snapshot_data, f, ensure_ascii=False, indent=2)
    print(f"리밸런싱 스냅샷 저장: {snap_path}")

    # 선정 종목 별도 CSV (분석 편의)
    if not selected.empty:
        sel_path = os.path.join(config.report_dir, f"selected_{yyyymmdd}.csv")
        selected.to_csv(sel_path, index=False, encoding="utf-8-sig")
        print(f"선정 종목 CSV 저장: {sel_path}")


def _month_index(dt: datetime) -> int:
    return dt.year * 12 + (dt.month - 1)


def _period_key(trading_date: str, rebalance_months: int, rebalance_days: int | None = None) -> str:
    if rebalance_days is not None and int(rebalance_days) > 0:
        dt = datetime.strptime(trading_date, "%Y-%m-%d")
        epoch = datetime(1970, 1, 1)
        days_since_epoch = (dt - epoch).days
        period_days = int(rebalance_days)
        period_idx = days_since_epoch // period_days
        period_start = epoch + timedelta(days=period_idx * period_days)
        return f"{period_start.strftime('%Y-%m-%d')}/{period_days}d"

    if rebalance_months <= 0:
        raise ValueError("rebalance_months must be positive")

    dt = datetime.strptime(trading_date, "%Y-%m-%d")
    date_idx = _month_index(dt)
    period_idx = date_idx // rebalance_months
    period_start_idx = period_idx * rebalance_months
    start_year = period_start_idx // 12
    start_month = (period_start_idx % 12) + 1
    return f"{start_year:04d}-{start_month:02d}/{rebalance_months}m"


def _load_run_state(path: str) -> dict[str, Any]:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as exc:
        print(f"[WARN] 상태 파일 로드 실패: {exc}")
    return {}


def _save_run_state(path: str, state: dict[str, Any]) -> None:
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _acquire_run_lock(lock_path: str):
    parent_dir = os.path.dirname(lock_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    lock_file = open(lock_path, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        raise RuntimeError(f"이미 실행 중입니다. lock={lock_path}")
    return lock_file


def _has_filled_rows_in_daily_report(config: LiveTradingConfig, trading_date: str) -> bool:
    if not trading_date:
        return False

    yyyymmdd = trading_date.replace("-", "")
    path = os.path.join(config.report_dir, f"fills_{yyyymmdd}.csv")
    if not os.path.exists(path):
        return False

    try:
        df = pd.read_csv(path)
    except Exception:
        return False

    if df.empty:
        return False

    if "is_filled" in df.columns:
        normalized = (
            df["is_filled"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        if normalized.isin({"true", "1", "1.0"}).any():
            return True

    if "round" in df.columns:
        round_col = df["round"].astype(str).str.strip().str.lower()
        if (round_col == "immediate_fill").any():
            return True

    return False


def _decide_execution_action(
    config: LiveTradingConfig,
    *,
    period_key: str,
    force: bool,
    dry_run: bool,
) -> tuple[ExecutionAction, dict[str, Any]]:
    state = _load_run_state(config.run_state_path)

    if dry_run:
        print(f"[DRY RUN] 주기 가드 상태와 무관하게 시뮬레이션 실행: period={period_key}")
        return "full_rebalance", state

    if not config.rebalance_guard_enabled:
        return "full_rebalance", state

    if force:
        print(f"[GUARD] force=true, 주기 가드 우회: period={period_key}")
        return "full_rebalance", state

    last_period_key = str(state.get("last_period_key") or "")
    if last_period_key != period_key:
        return "full_rebalance", state

    execution_state = _resolve_execution_state_from_state(state)
    if execution_state and state.get("execution_state") != execution_state:
        state["execution_state"] = execution_state
        _save_run_state(config.run_state_path, state)
        print(f"[GUARD] 구버전 상태 자동 마이그레이션: execution_state={execution_state}")

    if execution_state == "SKIPPED":
        note = str(state.get("note") or "")
        trading_date = str(state.get("last_trading_date") or "")
        if (
            note == "all_orders_skipped_missing_price_or_quote_error"
            and _has_filled_rows_in_daily_report(config, trading_date)
        ):
            execution_state = "SUCCESS"
            state["execution_state"] = execution_state
            state["note"] = "auto_healed_from_skipped_with_fills"
            _save_run_state(config.run_state_path, state)
            print(
                "[GUARD] 체결 리포트 기반 상태 자동 보정: "
                f"period={period_key}, execution_state=SUCCESS"
            )

    if execution_state is None:
        print("[GUARD] execution_state를 판단할 수 없어 안전 모드로 스킵합니다.")
        return "skip", state

    if execution_state in {"SUCCESS", "SKIPPED"}:
        return "skip", state

    if execution_state in {"ORDER_SUBMITTED", "PARTIAL_PENDING"}:
        return "reconcile_only", state

    if execution_state == "FAILED_BEFORE_ORDER":
        return "full_rebalance", state

    return "skip", state


def _persist_execution_state(
    config: LiveTradingConfig,
    *,
    period_key: str,
    signal_date: str,
    trading_date: str,
    execution_state: ExecutionState,
    note: str | None = None,
) -> None:
    if not config.rebalance_guard_enabled:
        return

    current = _load_run_state(config.run_state_path)
    current.update(
        {
            "last_period_key": period_key,
            "last_signal_date": signal_date,
            "last_trading_date": trading_date,
            "execution_state": execution_state,
            "updated_at": _kst_now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    if note:
        current["note"] = note
    _save_run_state(config.run_state_path, current)
    print(f"[GUARD] 상태 저장: period={period_key}, execution_state={execution_state}")


def _build_portfolio_history_from_reports(report_dir: str) -> list[dict]:
    """일별 체결 리포트(fills_YYYYMMDD.csv)에서 포트폴리오 가치 히스토리를 구성한다.

    변동성 타게팅에 필요한 {'date': str, 'portfolio_value': float} 리스트를 반환한다.
    파일이 없거나 읽기 실패 시 빈 리스트를 반환한다 (fail-open).
    """
    if not os.path.isdir(report_dir):
        return []

    records: list[tuple[str, float]] = []
    try:
        for fname in sorted(os.listdir(report_dir)):
            if not (fname.startswith("fills_") and fname.endswith(".csv")):
                continue
            fpath = os.path.join(report_dir, fname)
            try:
                df = pd.read_csv(fpath)
                if "portfolio_value" in df.columns and not df.empty:
                    date_val = str(df["timestamp"].iloc[-1])[:10] if "timestamp" in df.columns else fname[6:14]
                    port_val = float(df["portfolio_value"].iloc[-1])
                    records.append((date_val, port_val))
            except Exception:
                continue
    except Exception:
        return []

    return [{"date": d, "portfolio_value": v} for d, v in records]


async def run_once(signal_date: str | None = None, *, force: bool = False, dry_run: bool = False) -> None:
    config = LiveTradingConfig.from_env()
    if dry_run:
        config.dry_run_enabled = True
    missing = config.validate()
    if missing:
        raise RuntimeError(
            "필수 환경변수가 비어 있습니다: "
            + ", ".join(missing)
            + " (.env 로드 또는 셸 export 상태를 확인하세요)"
        )

    apply_kiwoom_client_session_patch()

    print(
        f"[CONFIG] mode={config.mode}, account_no={'set' if bool(config.account_no) else 'empty'}, "
        f"dry_run={config.dry_run_enabled}"
    )
    signal_date = signal_date or _kst_date_string()

    lock_file = _acquire_run_lock(config.run_lock_path)
    period_key = ""
    trading_date = signal_date
    state_written = False
    state_before_order: ExecutionState = "FAILED_BEFORE_ORDER"
    try:
        # LiveTradingConfig는 StrategyConfig를 상속하므로 직접 전달 가능
        signal_engine = LiveSignalEngine(config)

        signal = build_rebalance_signal(signal_engine, signal_date)
        trading_date = signal.trading_date
        period_key = _period_key(signal.trading_date, config.rebalance_months, config.rebalance_days)

        action, current_state = _decide_execution_action(
            config,
            period_key=period_key,
            force=force,
            dry_run=config.dry_run_enabled,
        )

        if action == "skip":
            print(
                f"[GUARD] 실행 스킵: trading_date={signal.trading_date}, period={period_key}, "
                f"execution_state={current_state.get('execution_state')}, state={config.run_state_path}"
            )
            return

        if action == "reconcile_only":
            print(
                f"[GUARD] 동일 주기 미완료 상태 감지. 미체결 주문 정리 후 reconcile_only 완료: "
                f"period={period_key}, execution_state={current_state.get('execution_state')}"
            )
            report_rows: list[dict] = []
            async with KiwoomBrokerAdapter(config) as broker:
                await _wait_until_order_time(config)
                all_resolved = await _reconcile_pending_orders(
                    broker=broker,
                    report_rows=report_rows,
                    trading_date=signal.trading_date,
                )
            _save_daily_report(config, signal.trading_date, report_rows)
            new_state: ExecutionState = "SUCCESS" if all_resolved else "PARTIAL_PENDING"
            _persist_execution_state(
                config,
                period_key=period_key,
                signal_date=signal_date,
                trading_date=signal.trading_date,
                execution_state=new_state,
                note=f"reconcile_only: {'all_resolved' if all_resolved else 'some_pending_remain'}",
            )
            state_written = True
            return

        if not config.dry_run_enabled:
            _persist_execution_state(
                config,
                period_key=period_key,
                signal_date=signal_date,
                trading_date=signal.trading_date,
                execution_state="STARTED",
            )

        if signal.selected.empty:
            print(f"[{signal.trading_date}] 선정 종목 없음")
            if not config.dry_run_enabled:
                _persist_execution_state(
                    config,
                    period_key=period_key,
                    signal_date=signal_date,
                    trading_date=signal.trading_date,
                    execution_state="SKIPPED",
                    note="empty_selection",
                )
            state_written = True
            return

        async with KiwoomBrokerAdapter(config) as broker:
            if config.dry_run_enabled:
                print("[DRY RUN] 주문 시각 대기/실주문/체결확인/상태저장을 수행하지 않습니다.")
            else:
                await _wait_until_order_time(config)
                # force 재실행이거나 이전 주기 상태로 진입한 경우 잔여 미체결 주문을 먼저 정리
                if force:
                    print("[SWEEP] force=True: 기존 미체결 주문 전량 취소 시작")
                    await _sweep_cancel_open_orders(broker)

            snapshot = await broker.get_account_snapshot()
            print(f"[DEBUG] 계정 스냅샷: 보유={snapshot.holdings}, 현금={snapshot.cash:,.0f}원")

            # 변동성 타게팅: 일별 리포트(fills)에서 포트폴리오 가치 히스토리를 구성해 유효 투자비율 계산
            effective_investment_ratio = config.investment_ratio
            _vol_multiplier: float | None = None
            _mt_in_market: bool | None = None
            _mt_reason: str | None = None
            if config.vol_target_enabled:
                try:
                    port_history = _build_portfolio_history_from_reports(config.report_dir)

                    # fills 데이터가 lookback에 미달할 때 pykrx 워밍업 히스토리로 보완
                    if (
                        config.vol_target_warmup_enabled
                        and len(port_history) < config.vol_target_lookback + 1
                    ):
                        _warmup_tickers = (
                            list(signal.selected["ticker"])
                            if "ticker" in signal.selected.columns
                            else []
                        )
                        if _warmup_tickers:
                            try:
                                warmup_hist = warmup_vol_history(
                                    _warmup_tickers,
                                    signal_date=signal.trading_date,
                                    lookback=config.vol_target_lookback,
                                )
                                if warmup_hist:
                                    # 동일 날짜 중복을 피하기 위해 warmup + fills 히스토리를 병합 후 최신 데이터 우선 유지
                                    merged = warmup_hist + port_history
                                    dedup_by_date: dict[str, dict] = {}
                                    for row in merged:
                                        date_key = str(row.get("date") or "")
                                        if date_key:
                                            dedup_by_date[date_key] = row
                                    port_history = [dedup_by_date[k] for k in sorted(dedup_by_date.keys())]
                            except Exception:
                                pass

                    vol_decision = compute_vol_target_ratio(
                        port_history,
                        base_ratio=config.investment_ratio,
                        enabled=config.vol_target_enabled,
                        sigma_target=config.vol_target_sigma,
                        lookback_days=config.vol_target_lookback,
                        min_ratio=config.vol_target_min_ratio,
                    )
                    effective_investment_ratio = vol_decision.effective_ratio
                    _vol_multiplier = vol_decision.multiplier
                    print(
                        f"[VOL-TARGET] reason={vol_decision.reason}, "
                        f"σ_realized={f'{vol_decision.sigma_realized*100:.1f}%' if vol_decision.sigma_realized is not None else 'N/A'}, "
                        f"σ_target={vol_decision.sigma_target*100:.0f}%, "
                        f"multiplier={vol_decision.multiplier:.3f}, "
                        f"base={vol_decision.base_ratio:.4f} → effective={effective_investment_ratio:.4f}"
                    )
                except Exception as _vol_err:
                    print(f"[VOL-TARGET] 계산 실패, base_ratio 유지: {_vol_err}")

            # 마켓 타이밍: KOSPI 200일 MA 기반 약세장 감지 시 투자비율 축소
            if config.market_timing_enabled:
                from market_timing import check_market_timing
                mt_decision = check_market_timing(
                    date_str=signal.trading_date,
                    enabled=config.market_timing_enabled,
                    ma_days=config.market_timing_ma_days,
                    cash_ratio_when_bearish=config.market_timing_cash_ratio,
                )
                print(
                    f"[MARKET-TIMING] reason={mt_decision.reason}, "
                    f"in_market={mt_decision.in_market}, "
                    f"multiplier={mt_decision.investment_ratio_multiplier:.2f}"
                )
                effective_investment_ratio = effective_investment_ratio * mt_decision.investment_ratio_multiplier
                _mt_in_market = mt_decision.in_market
                _mt_reason = mt_decision.reason
                if not mt_decision.in_market:
                    print(
                        f"[MARKET-TIMING] 약세장 감지 → "
                        f"투자비율 → {effective_investment_ratio:.4f}"
                    )

            if config.debug_signal_enabled:
                _print_selected_debug(signal.selected, max(1, config.debug_max_rows))

            selected_for_order = signal.selected
            if config.capital_constrained_selection_enabled:
                constrained_selected, alloc_meta = select_capital_constrained_stocks(
                    selected=signal.selected,
                    holdings=snapshot.holdings,
                    cash=snapshot.cash,
                    investment_ratio=effective_investment_ratio,
                    commission_fee_rate=config.commission_fee_rate,
                    max_stocks=config.capital_constrained_max_stocks,
                    min_stocks=config.capital_constrained_min_stocks,
                    slippage_rate=0.0,
                    holding_prices=snapshot.holding_prices,
                    existing_positions_policy=config.existing_positions_policy,
                )
                selected_for_order = constrained_selected
                print(
                    "[ALLOC] 자본제약 적용: "
                    f"before={int(alloc_meta['selected_before'])}, "
                    f"after={int(alloc_meta['selected_after'])}, "
                    f"k={int(alloc_meta['k_chosen'])}, "
                    f"주식당={float(alloc_meta['per_stock_amount']):,.0f}원"
                )

            intents = build_order_intents(
                selected=selected_for_order,
                holdings=snapshot.holdings,
                cash=snapshot.cash,
                investment_ratio=effective_investment_ratio,
                commission_fee_rate=config.commission_fee_rate,
                existing_positions_policy=config.existing_positions_policy,
                holding_prices=snapshot.holding_prices,
            )

            # 리밸런싱 시점 스냅샷 저장 (성과 분석·백테스트 비교용)
            _save_rebalance_snapshot(
                config=config,
                trading_date=signal.trading_date,
                selected=selected_for_order,
                holdings=snapshot.holdings,
                cash=snapshot.cash,
                effective_investment_ratio=effective_investment_ratio,
                vol_multiplier=_vol_multiplier,
                mt_in_market=_mt_in_market,
                mt_reason=_mt_reason,
            )

            if config.debug_signal_enabled:
                _print_intents_debug(
                    intents,
                    max_rows=max(1, config.debug_max_rows),
                    order_price_offset_bps=config.order_price_offset_bps,
                )

            if not intents:
                print(f"[DEBUG] 선정({len(selected_for_order)}개)되었으나 주문 의도 생성 실패")
                total_asset = snapshot.cash + sum(
                    snapshot.holdings.get(row.ticker, 0) * float(row.close)
                    for row in selected_for_order.itertuples(index=False)
                )
                invest_amount = total_asset * effective_investment_ratio
                per_stock_amount = invest_amount / len(selected_for_order) if len(selected_for_order) > 0 else 0
                print(f"[DEBUG] 총자산={total_asset:,.0f}원, 투자금={invest_amount:,.0f}원, 주식당={per_stock_amount:,.0f}원")
                print(f"[{signal.trading_date}] 주문 대상 없음")
                if not config.dry_run_enabled:
                    _persist_execution_state(
                        config,
                        period_key=period_key,
                        signal_date=signal_date,
                        trading_date=signal.trading_date,
                        execution_state="SKIPPED",
                        note="no_intent",
                    )
                state_written = True
                return

            print(f"[{signal.trading_date}] 주문 생성: {len(intents)}건")

            # type=00은 item 무관하게 계좌 전체 주문 이벤트를 수신 — item=[] 단일 등록으로 충분
            try:
                grp_no = f"pre_orders_{int(time.time() * 1000)}"
                await broker.register_real_for_orders(grp_no, [], refresh="1")
                print(f"[WS] pre-registered type=00 account-level subscription (grp={grp_no})")
            except Exception as exc:
                print(f"[WARN] pre-register REAL unexpected error: {exc}")
            if config.dry_run_enabled:
                for intent in intents:
                    limit_price = _limit_price(intent.side, intent.reference_price, config.order_price_offset_bps)
                    # DRY RUN 미리보기용으로 제출 전 라운딩을 적용합니다
                    try:
                        explicit_tick = None
                        if config.use_api_tick_when_available:
                            try:
                                quote = await broker.get_best_quote(intent.ticker)
                                explicit_tick = getattr(quote, "tick_size", None)
                            except Exception:
                                explicit_tick = None
                        if explicit_tick and int(explicit_tick) > 0:
                            tick = int(explicit_tick)
                            rem = int(limit_price) % tick
                            if rem * 2 < tick:
                                adj = int(limit_price) - rem
                            else:
                                adj = int(limit_price) + (tick - rem)
                        else:
                            adj = broker.round_price_to_tick(int(limit_price), mode="nearest")
                    except Exception:
                        adj = int(limit_price)

                    print(
                        f"[DRY RUN] order ticker={intent.ticker} side={intent.side} qty={intent.quantity} "
                        f"price={adj} reason={intent.reason}"
                    )
                print(f"[DRY RUN] 시뮬레이션 완료: planned_orders={len(intents)}")
                state_written = True
                return

            submitted_orders = []
            report_rows: list[dict] = []
            # 주문 제출 직후 즉시 WS 체결 이벤트를 등록하기 위해 루프 전에 초기화합니다.
            # (제출-대기 구간 중 도착하는 체결 이벤트를 누락 없이 포착)
            fill_events: dict[str, asyncio.Event] = {}
            for intent in intents:
                limit_price = _limit_price(intent.side, intent.reference_price, config.order_price_offset_bps)

                # 보조 가격 보완: 계산된 limit_price가 0인 경우 시세 조회로 대체
                if limit_price <= 0:
                    try:
                        quote = await broker.get_best_quote(intent.ticker)
                        fallback = quote.ask1 if intent.side == "BUY" else quote.bid1
                        if not fallback or fallback <= 0:
                            # 시세로 보완 불가
                            print(f"[WARN] 시세로 보완 불가(호가 없음): ticker={intent.ticker} side={intent.side}; 주문 스킵")
                            # 기록 후 스킵
                            report_rows.append(
                                {
                                    "timestamp": _kst_now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "round": "pre_submit_skip",
                                    "ticker": intent.ticker,
                                    "side": intent.side,
                                    "order_no": "",
                                    "order_price": None,
                                    "requested_qty": intent.quantity,
                                    "pending_qty": intent.quantity,
                                    "is_filled": False,
                                    "return_code": None,
                                    "note": "missing_price",
                                }
                            )
                            print(f"[ORDER] skip ticker={intent.ticker} side={intent.side} qty={intent.quantity} reason=missing_price")
                            continue

                        print(f"[ORDER] limit_price==0 보완: ticker={intent.ticker} side={intent.side} fallback_price={fallback}")
                        limit_price = fallback
                    except Exception as exc:
                        print(f"[WARN] 시세 조회로 가격 보완 실패: {exc}; ticker={intent.ticker}; 주문 스킵")
                        report_rows.append(
                            {
                                "timestamp": _kst_now().strftime("%Y-%m-%d %H:%M:%S"),
                                "round": "pre_submit_skip",
                                "ticker": intent.ticker,
                                "side": intent.side,
                                "order_no": "",
                                "order_price": None,
                                "requested_qty": intent.quantity,
                                "pending_qty": intent.quantity,
                                "is_filled": False,
                                "return_code": None,
                                "note": "quote_fetch_error",
                            }
                        )
                        print(f"[ORDER] skip ticker={intent.ticker} side={intent.side} qty={intent.quantity} reason=quote_fetch_error")
                        continue

                try:
                    # 구성된 경우 API에서 제공하는 틱 사이즈를 조회하여 submit_order에 전달합니다
                    explicit_tick = None
                    try:
                        if config.use_api_tick_when_available:
                            try:
                                quote = await broker.get_best_quote(intent.ticker)
                                explicit_tick = getattr(quote, "tick_size", None)
                            except Exception:
                                explicit_tick = None
                    except Exception:
                        explicit_tick = None

                    submitted = await broker.submit_order(
                        ticker=intent.ticker,
                        side=intent.side,
                        quantity=intent.quantity,
                        price=limit_price,
                        explicit_tick=explicit_tick,
                    )
                except KiwoomAPIError as exc:
                    # 디버깅/재현을 용이하게 하기 위해 전체 API 응답을 로그합니다
                    try:
                        dumped = json.dumps(exc.body or {}, ensure_ascii=False)
                        print(f"[ORDER][KIWOOM_ERROR] api_id={exc.api_id} endpoint={exc.endpoint} body={dumped}")
                    except Exception:
                        print(f"[ORDER][KIWOOM_ERROR] api_id={exc.api_id} endpoint={exc.endpoint} body={exc.body}")

                    # 모의 서버가 모의투자 장종료를 신호하면(예: return_code==20) 주문을 안전하게 건너뜁니다
                    try:
                        rc = int(exc.return_code) if exc.return_code is not None else None
                    except Exception:
                        rc = None

                    if rc == 20:
                        # '장마감' 오류와 '호가단위' 오류를 구분합니다.
                        # 일부 Kiwoom 모의 응답은 return_code==20을 사용하지만 메시지에
                        # RC4003이나 '호가단위'가 포함되어 있어 틱 단위 문제를 나타낼 수 있습니다.
                        msg = str(getattr(exc, "return_msg", "") or "")
                        body_msg = ""
                        try:
                            if isinstance(getattr(exc, "body", None), dict):
                                body_msg = str((exc.body or {}).get("return_msg") or "")
                        except Exception:
                            body_msg = ""

                        combined = f"{msg} {body_msg}".strip()
                        if "RC4003" in combined or "호가단위" in combined or "호가 단위" in combined:
                            print(f"[ORDER] 모의투자 호가단위 오류 감지: ticker={intent.ticker} side={intent.side} return_code={rc}; 라운딩 후 재시도 시도")

                            # 유효한 틱 단위로 반올림하여 구성된 횟수만큼 재전송을 시도합니다
                            try:
                                # 반올림 모드 결정: 매수는 반올림(nearest), 매도도 nearest로 처리
                                mode = "nearest"
                                adj_price = broker.round_price_to_tick(limit_price, mode=mode)
                                                            # 조정된 가격이 변하지 않으면 상/하 보완 로직 시도
                                if adj_price == int(limit_price):
                                    adj_price_down = broker.round_price_to_tick(limit_price, mode="down")
                                    adj_price_up = broker.round_price_to_tick(limit_price, mode="up")
                                    # 우선 내림(down) 시도 후 올림(up) 시도
                                    adj_price = adj_price_down or adj_price_up or adj_price

                                # 조정된 가격이 양수이며 원가와 다른 경우에만 시도합니다
                                if adj_price and int(adj_price) > 0 and int(adj_price) != int(limit_price):
                                    attempt_rounds = 0
                                    max_rounds = int(getattr(config, "max_retry_rounds", 2) or 2)
                                    retried = None
                                    while attempt_rounds < max_rounds:
                                        attempt_rounds += 1
                                        try:
                                            print(f"[ORDER] tick-round retry attempt={attempt_rounds}/{max_rounds}: original={limit_price} adj={adj_price}")
                                            retried = await broker.submit_order(
                                                ticker=intent.ticker,
                                                side=intent.side,
                                                quantity=intent.quantity,
                                                price=int(adj_price),
                                            )
                                            # 성공하면 리포트하고 추가 처리를 진행합니다
                                            submitted_orders.append(retried)
                                            # 제출 직후 즉시 WS 체결 이벤트 등록
                                            if retried and retried.order_no:
                                                fill_events[retried.order_no] = broker.register_fill_event(retried.order_no)
                                            report_rows.append(
                                                {
                                                    "timestamp": _kst_now().strftime("%Y-%m-%d %H:%M:%S"),
                                                    "round": "submit_retry_tick_round",
                                                    "ticker": intent.ticker,
                                                    "side": intent.side,
                                                    "order_no": retried.order_no if retried else "",
                                                    "order_price": int(adj_price),
                                                    "requested_qty": intent.quantity,
                                                    "pending_qty": intent.quantity,
                                                    "is_filled": False,
                                                    "return_code": None,
                                                    "note": "mock_tick_unit_retry",
                                                }
                                            )
                                            break
                                        except KiwoomAPIError as exc2:
                                            # 여전히 호가단위 오류이면 다음 조정을 시도하거나 중단합니다
                                            try:
                                                body_msg2 = str((exc2.body or {}).get("return_msg") or "")
                                            except Exception:
                                                body_msg2 = str(getattr(exc2, "return_msg", "") or "")
                                            if "RC4003" in body_msg2 or "호가단위" in body_msg2 or "호가 단위" in body_msg2:
                                                # 한 번 다른 방향으로 시도합니다
                                                if attempt_rounds == 1:
                                                    adj_price = broker.round_price_to_tick(limit_price, mode="down") or broker.round_price_to_tick(limit_price, mode="up")
                                                    continue
                                            # 다른 오류: 포기하고 외부 핸들러로 계속 진행합니다
                                            raise
                                else:
                                    print(f"[ORDER] 라운딩으로 유효가격 생성 불가: original={limit_price} adj={adj_price}; 스킵 처리")
                                    report_rows.append(
                                        {
                                            "timestamp": _kst_now().strftime("%Y-%m-%d %H:%M:%S"),
                                            "round": "submit_skip",
                                            "ticker": intent.ticker,
                                            "side": intent.side,
                                            "order_no": "",
                                            "order_price": limit_price,
                                            "requested_qty": intent.quantity,
                                            "pending_qty": intent.quantity,
                                            "is_filled": False,
                                            "return_code": rc,
                                            "note": "mock_tick_unit_error_no_valid_price",
                                        }
                                    )
                                    # 다음 주문 의도로 이동
                                    continue
                            except KiwoomAPIError:
                                # 상위 핸들러로 예외를 전달해 폴백 정책을 결정하도록 함
                                raise
                            except Exception as exc_round:
                                print(f"[ORDER] tick-round 재시도 중 예외: {exc_round}; 스킵 처리")
                                report_rows.append(
                                    {
                                        "timestamp": _kst_now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "round": "submit_skip",
                                        "ticker": intent.ticker,
                                        "side": intent.side,
                                        "order_no": "",
                                        "order_price": limit_price,
                                        "requested_qty": intent.quantity,
                                        "pending_qty": intent.quantity,
                                        "is_filled": False,
                                        "return_code": rc,
                                        "note": "mock_tick_unit_retry_failed",
                                    }
                                )
                                continue

                        # 폴백: 이전과 동일하게 모의(Mock)에서 장마감으로 처리
                        print(f"[ORDER] 모의투자 장종료 감지: ticker={intent.ticker} side={intent.side} return_code={rc}; 스킵 처리")
                        report_rows.append(
                            {
                                "timestamp": _kst_now().strftime("%Y-%m-%d %H:%M:%S"),
                                "round": "submit_skip",
                                "ticker": intent.ticker,
                                "side": intent.side,
                                "order_no": "",
                                "order_price": limit_price,
                                "requested_qty": intent.quantity,
                                "pending_qty": intent.quantity,
                                "is_filled": False,
                                "return_code": rc,
                                "note": "mock_market_closed",
                            }
                        )
                                    # 다음 주문 의도로 이동
                        continue

                    # 다른 키움 오류의 경우, 설정된 폴백 코드와 대조하여 일치하면 시장가로 재시도합니다
                    fallback_codes = tuple(config.fallback_to_market_return_codes or ())
                    should_fallback = False
                    try:
                        if broker._kerr_matches_codes(exc, fallback_codes):
                            should_fallback = True
                    except Exception:
                        # 메시지/코드 기반의 간단한 휴리스틱 매칭으로 폴백
                        msg = str(exc.return_msg or "")
                        for c in fallback_codes:
                            try:
                                if rc == int(c):
                                    should_fallback = True
                                    break
                            except Exception:
                                pass
                            if f"RC{c}" in msg or str(c) in msg:
                                should_fallback = True
                                break

                    if should_fallback:
                        submitted = await broker.submit_order(
                            ticker=intent.ticker,
                            side=intent.side,
                            quantity=intent.quantity,
                            price=limit_price,
                            order_type="3",
                        )
                    else:
                        # 예기치 않은 오류는 외부 핸들러에 위임(재발생)하여 오류를 노출시킵니다
                        raise
                # 임시로 추가합니다; 아래 배치 검사에서 즉시 체결된 주문은 제거됩니다
                submitted_orders.append(submitted)
                # 제출 직후 즉시 WS 체결 이벤트 등록 — 다음 주문 제출 중 도착하는 체결 이벤트를 포착
                if submitted.order_no:
                    fill_events[submitted.order_no] = broker.register_fill_event(submitted.order_no)
                print(
                    f"order ticker={intent.ticker} side={intent.side} qty={intent.quantity} "
                    f"price={limit_price} order_no={submitted.order_no} return_code={submitted.raw_response.get('return_code')}"
                )

                # 주문 간 짧은 지연을 두어 레이트리밋 완화
                try:
                    await asyncio.sleep(max(0.0, float(config.order_submit_delay_seconds)))
                except Exception:
                    pass

            # 제출 자체가 없던 경우(all skipped)와
            # 제출 후 즉시 전량 체결된 경우를 구분하기 위한 기준값
            initial_submitted_count = len(submitted_orders)

            # ── Batch fill monitoring: register fill events for all submitted orders,
            # then wait for WS signals + a single batch poll ──
            try:
                initial_wait = float(getattr(config, "order_fill_initial_wait_seconds", 5.0) or 5.0)
            except Exception:
                initial_wait = 5.0

            if submitted_orders:
                # fill_events는 루프 중 각 submit 직후에 이미 등록됨
                # — 여기서는 추가 등록 없이 WS 기반 즉시 체결을 위해 짧게 대기합니다

                # WS 기반 즉시 체결을 위해 잠시 대기합니다
                if fill_events:
                    try:
                        await asyncio.sleep(min(initial_wait, 2.0))
                    except Exception:
                        pass

                # WS가 놓친 항목을 포착하기 위한 단일 배치 폴
                batch_checks: list = []
                try:
                    batch_checks = await broker.check_orders_batch(submitted_orders)
                    for bc in batch_checks:
                        if bc.is_filled:
                            try:
                                submitted_orders = [s for s in submitted_orders if s.order_no != bc.submitted.order_no]
                                report_rows.append(
                                    {
                                        "timestamp": _kst_now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "round": "immediate_fill",
                                        "ticker": bc.submitted.ticker,
                                        "side": bc.submitted.side,
                                        "order_no": bc.submitted.order_no,
                                        "order_price": bc.submitted.price,
                                        "requested_qty": bc.submitted.requested_qty,
                                        "pending_qty": 0,
                                        "is_filled": True,
                                        "return_code": bc.submitted.raw_response.get("return_code"),
                                        "note": "immediate_fill_batch_poll",
                                    }
                                )
                            except Exception:
                                pass
                    filled_count = sum(1 for bc in batch_checks if bc.is_filled)
                    pending_count = len(batch_checks) - filled_count
                    print(f"[BATCH] initial check: filled={filled_count}, pending={pending_count}")
                except Exception as exc:
                    print(f"[WARN] batch initial check failed: {exc}")

                # 이미 체결 확인된 주문만 이벤트 해제.
                # 미체결 주문의 이벤트는 그대로 유지 → wait_for_fill_window가 기존 이벤트를
                # 재사용하므로 배치 체크~wait_for_fill_window 진입 사이 도착하는 WS 체결을 누락하지 않음.
                try:
                    filled_order_nos = {bc.submitted.order_no for bc in batch_checks if bc.is_filled}
                except Exception:
                    filled_order_nos = set()
                for order_no in list(fill_events.keys()):
                    if order_no in filled_order_nos:
                        try:
                            broker.unregister_fill_event(order_no)
                        except Exception:
                            pass

            if not submitted_orders and initial_submitted_count == 0:
                print(f"[GUARD] 모든 주문이 스킵되었습니다. submitted_orders=0")
                if not config.dry_run_enabled:
                    _persist_execution_state(
                        config,
                        period_key=period_key,
                        signal_date=signal_date,
                        trading_date=signal.trading_date,
                        execution_state="SKIPPED",
                        note="all_orders_skipped_missing_price_or_quote_error",
                    )
                state_written = True
                _save_daily_report(config, signal.trading_date, report_rows)
                return

            # 주문 제출은 있었지만 초기 배치 확인에서 모두 체결된 경우
            if not submitted_orders and initial_submitted_count > 0:
                print(f"[BATCH] 초기 배치 확인에서 제출 주문 {initial_submitted_count}건이 모두 체결되었습니다.")
                _save_daily_report(config, signal.trading_date, report_rows)
                _persist_execution_state(
                    config,
                    period_key=period_key,
                    signal_date=signal_date,
                    trading_date=signal.trading_date,
                    execution_state="SUCCESS",
                    note=f"filled_in_initial_batch={initial_submitted_count}",
                )
                state_written = True
                return

            _persist_execution_state(
                config,
                period_key=period_key,
                signal_date=signal_date,
                trading_date=signal.trading_date,
                execution_state="ORDER_SUBMITTED",
                note=f"submitted_orders={len(submitted_orders)}",
            )
            state_before_order = "PARTIAL_PENDING"

            current_orders = submitted_orders
            final_pending = 0
            for round_idx in range(1, max(0, config.max_retry_rounds) + 1):
                print(f"미체결 확인 대기: {config.order_timeout_minutes}분 (round={round_idx})")
                await broker.wait_for_fill_window(config.order_timeout_minutes, current_orders)

                retried_orders, checks = await broker.run_retry_cycle(
                    submitted_orders=current_orders,
                )

                _append_report_rows(report_rows, checks, round_name=f"retry_check_{round_idx}")

                filled = sum(1 for c in checks if c.is_filled)
                pending = len(checks) - filled
                final_pending = pending
                print(f"{round_idx}차 체결 결과: filled={filled}, pending={pending}")

                if pending == 0:
                    current_orders = []
                    break

                for check in checks:
                    if check.is_filled:
                        continue
                    print(
                        f"retry ticker={check.submitted.ticker} side={check.submitted.side} "
                        f"pending_qty={check.pending_qty}"
                    )

                if retried_orders:
                    print(f"{round_idx}차 재주문 전송: {len(retried_orders)}건")

                current_orders = retried_orders

            final_checks = []
            if current_orders:
                print(f"최종 체결 확인 대기: {config.order_timeout_minutes}분")
                await broker.wait_for_fill_window(config.order_timeout_minutes, current_orders)
                final_checks = await broker.check_orders(current_orders)
                _append_report_rows(report_rows, final_checks, round_name="final_check")

                final_filled = sum(1 for c in final_checks if c.is_filled)
                final_pending = len(final_checks) - final_filled
                print(f"최종 체결 확인: filled={final_filled}, pending={final_pending}")

            _save_daily_report(config, signal.trading_date, report_rows)
            if final_pending > 0:
                _persist_execution_state(
                    config,
                    period_key=period_key,
                    signal_date=signal_date,
                    trading_date=signal.trading_date,
                    execution_state="PARTIAL_PENDING",
                    note=f"pending_qty_exists_after_final_check={final_pending}",
                )
            else:
                _persist_execution_state(
                    config,
                    period_key=period_key,
                    signal_date=signal_date,
                    trading_date=signal.trading_date,
                    execution_state="SUCCESS",
                )
            state_written = True
    except Exception as exc:
        if config.rebalance_guard_enabled and period_key and not state_written:
            _persist_execution_state(
                config,
                period_key=period_key,
                signal_date=signal_date,
                trading_date=trading_date,
                execution_state=state_before_order,
                note=f"exception={type(exc).__name__}: {exc}",
            )
        raise
    finally:
        lock_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live rebalance runner")
    parser.add_argument("--signal-date", type=str, default=None, help="신호 기준일 (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="이미 실행한 주기라도 강제로 실행")
    parser.add_argument("--dry-run", action="store_true", help="주문 없이 시뮬레이션만 실행")
    args = parser.parse_args()

    asyncio.run(run_once(signal_date=args.signal_date, force=args.force, dry_run=args.dry_run))
