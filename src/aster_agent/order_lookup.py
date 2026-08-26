"""Order-status lookup tool.

This is a pure, deterministic function — no LLM involved — so its safety
properties are unit-testable in isolation. It enforces every rule from
`data/orders-data-dictionary.md`:

* Input normalization (uppercase, strip whitespace/surrounding punctuation) but
  never guessing a different order ID.
* A strict allow-list of customer-safe fields. Customer PII and everything under
  `internal` can never leave this function.
* `status` is authoritative. Stale carrier/tracking/ETA fields are suppressed for
  cancelled and returned orders.
* `shipped` with a null ETA reports "estimate unavailable" rather than inventing
  a date.
* `exception` status flags that human review is required.
* Unknown / malformed IDs return a safe not-found result, never a fabricated one.

The tool returns a structured dict. `found` and `error` let the agent tell,
deterministically, whether a real lookup happened.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import ORDERS_PATH

# Fields safe to surface to the customer / model context.
_CUSTOMER_SAFE_FIELDS = (
    "order_id",
    "membership_tier",
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
)
_SAFE_ITEM_FIELDS = ("name", "quantity", "final_sale")

# Statuses for which carrier/tracking/ETA fields are stale and must be hidden.
_TERMINAL_NO_DELIVERY = {"cancelled", "returned"}

_ORDER_ID_RE = re.compile(r"^ORD-\d+$")


class OrderStore:
    """Loads orders once and serves sanitized lookups."""

    def __init__(self, orders_path: Path | None = None) -> None:
        self.path = orders_path or ORDERS_PATH
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.snapshot_at: str = raw.get("snapshot_at", "")
        self._orders: dict[str, dict[str, Any]] = {
            o["order_id"].upper(): o for o in raw.get("orders", [])
        }

    @staticmethod
    def normalize_id(raw_id: str) -> str:
        """Normalize harmless input differences without guessing a new ID.

        Uppercases, strips whitespace and surrounding punctuation. Does not
        attempt fuzzy matching or digit correction.
        """
        if not isinstance(raw_id, str):
            return ""
        cleaned = raw_id.strip().upper()
        # Strip surrounding ordinary punctuation/quotes but keep the internal hyphen.
        cleaned = cleaned.strip(".,;:!?\"'()[]{}<>")
        cleaned = cleaned.replace(" ", "")
        return cleaned

    def lookup(self, order_id: str) -> dict[str, Any]:
        normalized = self.normalize_id(order_id)

        if not normalized:
            return {
                "found": False,
                "error": "missing_order_id",
                "message": "No order ID was provided.",
                "requires_human": False,
            }

        if not _ORDER_ID_RE.match(normalized):
            return {
                "found": False,
                "error": "malformed_order_id",
                "normalized_order_id": normalized,
                "message": f"'{order_id}' is not a valid order ID format (expected ORD-####).",
                "requires_human": False,
            }

        order = self._orders.get(normalized)
        if order is None:
            return {
                "found": False,
                "error": "not_found",
                "normalized_order_id": normalized,
                "message": f"No order was found for {normalized}.",
                "requires_human": True,
            }

        return self._sanitize(order)

    def _sanitize(self, order: dict[str, Any]) -> dict[str, Any]:
        status = str(order.get("status", "")).lower()

        safe: dict[str, Any] = {"found": True, "error": None}
        for field in _CUSTOMER_SAFE_FIELDS:
            if field in order:
                safe[field] = order[field]

        safe["items"] = [
            {k: item.get(k) for k in _SAFE_ITEM_FIELDS if k in item}
            for item in order.get("items", [])
        ]

        # Status precedence: suppress stale delivery fields for terminal states.
        if status in _TERMINAL_NO_DELIVERY:
            safe["carrier"] = None
            safe["tracking_number"] = None
            safe["estimated_delivery"] = None
            safe["delivery_estimate_available"] = False
        else:
            safe["delivery_estimate_available"] = order.get("estimated_delivery") is not None

        # shipped + no ETA: explicit signal so the agent doesn't invent a date.
        if status == "shipped" and not order.get("estimated_delivery"):
            safe["delivery_estimate_available"] = False

        # exception: requires human review.
        safe["requires_human"] = status == "exception"

        return safe


# Module-level singleton for convenience; the agent passes this through as a tool.
_DEFAULT_STORE: OrderStore | None = None


def get_store() -> OrderStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = OrderStore()
    return _DEFAULT_STORE


def order_lookup(order_id: str) -> dict[str, Any]:
    """Public tool entrypoint used by the agent's function-calling loop."""
    return get_store().lookup(order_id)
