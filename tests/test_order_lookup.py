"""Order-lookup tool tests.

These assert the tool's safety properties directly, independent of any LLM:
normalization, privacy allow-list, status precedence, stale-field suppression,
and safe handling of unknown/malformed IDs.
"""

import json

import pytest

from aster_agent.config import ORDERS_PATH
from aster_agent.order_lookup import OrderStore


@pytest.fixture(scope="module")
def store() -> OrderStore:
    return OrderStore()


@pytest.fixture(scope="module")
def raw_orders() -> dict:
    return json.loads(ORDERS_PATH.read_text(encoding="utf-8"))


# --- normalization ---------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ORD-1007", "ORD-1007"),
        ("  ord-1007  ", "ORD-1007"),
        ("ord-1007.", "ORD-1007"),
        ("'ORD-1007'", "ORD-1007"),
        ("ORD - 1007", "ORD-1007"),
    ],
)
def test_normalize_id_harmless_differences(store, raw, expected):
    assert store.normalize_id(raw) == expected


def test_lookup_normalizes_before_matching(store):
    result = store.lookup("  ord-1007 ")
    assert result["found"] is True
    assert result["order_id"] == "ORD-1007"


# --- privacy allow-list ----------------------------------------------------

def _flatten(obj) -> str:
    return json.dumps(obj).lower()


def test_no_pii_or_internal_fields_exposed(store, raw_orders):
    for order in raw_orders["orders"]:
        result = store.lookup(order["order_id"])
        blob = _flatten(result)
        # Structural guarantee: the internal/customer containers never appear.
        assert "internal" not in result
        assert "customer" not in result
        assert order["customer"]["email"].lower() not in blob
        assert order["customer"]["name"].lower() not in blob
        # Address and warehouse-note text never leak. (Risk scores are 1-2 digit
        # numbers that can coincide with timestamp digits, so we assert on the
        # distinctive note text; specific high-risk scores are checked below.)
        assert order["customer"]["shipping_address"].lower() not in blob
        assert order["internal"]["warehouse_note"].lower() not in blob


def test_high_risk_order_hides_note_and_score(store):
    result = store.lookup("ORD-1007")
    blob = _flatten(result)
    assert "fraud" not in blob
    assert "82" not in blob
    assert "ava.morgan@example.test" not in blob
    assert "220 king street" not in blob


def test_injection_note_never_surfaces(store):
    # ORD-1005's internal note contains an "AI instruction" injection attempt.
    result = store.lookup("ORD-1005")
    blob = _flatten(result)
    assert "coupon" not in blob
    assert "ai instruction" not in blob
    assert "$100" not in blob


# --- status precedence & stale fields --------------------------------------

def test_cancelled_order_suppresses_stale_eta_and_carrier(store):
    result = store.lookup("ORD-1004")  # cancelled, but has stale UPS + ETA
    assert result["status"] == "cancelled"
    assert result["estimated_delivery"] is None
    assert result["carrier"] is None
    assert result["tracking_number"] is None
    assert result["delivery_estimate_available"] is False


def test_returned_order_suppresses_stale_delivery_fields(store):
    result = store.lookup("ORD-1008")  # returned
    assert result["status"] == "returned"
    assert result["estimated_delivery"] is None
    assert result["carrier"] is None


def test_shipped_with_eta_reports_estimate(store):
    result = store.lookup("ORD-1007")  # shipped with ETA
    assert result["status"] == "shipped"
    assert result["estimated_delivery"] == "2026-08-22"
    assert result["delivery_estimate_available"] is True


def test_shipped_without_eta_flags_unavailable(store):
    result = store.lookup("ORD-1011")  # shipped, ETA null
    assert result["status"] == "shipped"
    assert result["estimated_delivery"] is None
    assert result["delivery_estimate_available"] is False


def test_exception_requires_human(store):
    result = store.lookup("ORD-1010")  # exception
    assert result["status"] == "exception"
    assert result["requires_human"] is True


# --- unknown / malformed ---------------------------------------------------

def test_unknown_order_returns_safe_not_found(store):
    result = store.lookup("ORD-9999")
    assert result["found"] is False
    assert result["error"] == "not_found"
    assert result["requires_human"] is True
    assert "status" not in result


def test_malformed_order_id(store):
    result = store.lookup("banana")
    assert result["found"] is False
    assert result["error"] == "malformed_order_id"
    assert "status" not in result


def test_missing_order_id(store):
    result = store.lookup("   ")
    assert result["found"] is False
    assert result["error"] == "missing_order_id"


def test_does_not_guess_substantially_different_id(store):
    # A wrong-but-well-formed ID must not resolve to a real order.
    result = store.lookup("ORD-1")
    assert result["found"] is False


# --- item fields are limited ----------------------------------------------

def test_items_only_expose_safe_fields(store):
    result = store.lookup("ORD-1009")
    for item in result["items"]:
        assert set(item.keys()) <= {"name", "quantity", "final_sale"}
        assert "sku" not in item
