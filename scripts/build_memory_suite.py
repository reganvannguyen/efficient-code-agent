"""Build a 5-task multi-file Python lookup & working-memory benchmark suite.

Each task is an isolated local git repository with 12-14 Python modules
(domain rules spread across 4 modules plus realistic supporting/decoy modules)
so that:
  1. Prompts describe only the observable bug (no file paths named), testing
     repository lookup.
  2. Fixing each bug requires combining rules/constants scattered across
     multiple files while total repo size pushes past an 8K context window.
  3. Every repository has a deterministic pytest suite (`oracle_tier: local_full`).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from harness.evaluation import load_suite_dir


@dataclass(frozen=True)
class TaskSpec:
    slug: str
    task_id: str
    difficulty: str
    issue_text: str
    test_file: str
    files: dict[str, str]
    reference_overrides: dict[str, str]


def _make_decoy_module(module_name: str, domain: str, seed: int) -> str:
    """Generate a realistic ~140-line supporting Python module for repo scale."""
    lines = [
        f'"""Supporting {module_name} utilities for the {domain} package."""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass, field",
        "from typing import Any",
        "",
        f"MODULE_VERSION = '1.{seed}.0'",
        f"DEFAULT_BATCH_SIZE = {25 * (seed + 1)}",
        f"MAX_RECORDS_PER_EXPORT = {500 * (seed + 1)}",
        "",
        "",
        "@dataclass(frozen=True)",
        f"class {module_name.title().replace('_', '')}Record:",
        f'    """Serialized {module_name} entry for {domain} reporting."""',
        "",
        "    record_id: str",
        "    category: str",
        "    payload: dict[str, Any] = field(default_factory=dict)",
        "    version: str = MODULE_VERSION",
        "",
        "    def to_row(self) -> tuple[str, str, int]:",
        "        return (self.record_id, self.category, len(self.payload))",
        "",
        "    def matches_prefix(self, prefix: str) -> bool:",
        "        return self.record_id.startswith(prefix) or self.category.startswith(prefix)",
        "",
    ]
    for idx in range(1, 7):
        lines.extend(
            [
                "",
                f"def format_{module_name}_section_{idx}(",
                f"    records: list[{module_name.title().replace('_', '')}Record],",
                "    *,",
                "    include_empty: bool = False,",
                "    max_items: int = DEFAULT_BATCH_SIZE,",
                ") -> list[str]:",
                f'    """Render section {idx} summary lines for {domain} {module_name}."""',
                "    output: list[str] = []",
                "    for entry in records[:max_items]:",
                "        if not include_empty and not entry.payload:",
                "            continue",
                "        keys = ', '.join(sorted(entry.payload.keys()))",
                f'        output.append(f"[{module_name}:{idx}] {{entry.record_id}} ({{entry.category}}): {{keys}}")',
                "    return output",
            ]
        )
    lines.extend(
        [
            "",
            f"def summarize_{module_name}(",
            f"    records: list[{module_name.title().replace('_', '')}Record],",
            ") -> dict[str, int]:",
            f'    """Count {module_name} records by category."""',
            "    counts: dict[str, int] = {}",
            "    for entry in records:",
            "        counts[entry.category] = counts.get(entry.category, 0) + 1",
            "    return counts",
            "",
        ]
    )
    return "\n".join(lines)


def _common_repo_files(pkg: str, domain: str) -> dict[str, str]:
    files: dict[str, str] = {
        ".gitignore": "__pycache__/\n*.pyc\n.pytest_cache/\n.harness/\n",
        "conftest.py": (
            "import sys\n"
            "from pathlib import Path\n\n"
            "ROOT = str(Path(__file__).resolve().parent)\n"
            "if ROOT not in sys.path:\n"
            "    sys.path.insert(0, ROOT)\n"
        ),
        f"{pkg}/__init__.py": f'"""The {domain} service package."""\n',
        f"{pkg}/supporting/__init__.py": f'"""Supporting modules for {domain}."""\n',
    }
    decoys = [
        "audit_log",
        "csv_export",
        "cli_runner",
        "report_builder",
        "schema_validator",
        "telemetry",
        "migrations",
    ]
    for idx, name in enumerate(decoys):
        files[f"{pkg}/supporting/{name}.py"] = _make_decoy_module(name, domain, idx)
    return files


def _build_invoice_task() -> TaskSpec:
    pkg = "billing"
    files = _common_repo_files(pkg, "invoice billing")
    files.update(
        {
            f"{pkg}/rates.py": '''"""Regional sales tax rates for billing."""

from __future__ import annotations

from decimal import Decimal

# Standard tax rate by ISO region code.
REGIONAL_TAX_RATES: dict[str, Decimal] = {
    "US-CA": Decimal("0.0825"),
    "US-NY": Decimal("0.08875"),
    "US-OR": Decimal("0.00"),
    "EU-DE": Decimal("0.19"),
    "CA-ON": Decimal("0.13"),
}

FALLBACK_TAX_RATE = Decimal("0.05")


def get_tax_rate(region: str) -> Decimal:
    """Return the configured Decimal tax rate for a region."""
    return REGIONAL_TAX_RATES.get(region.upper(), FALLBACK_TAX_RATE)
''',
            f"{pkg}/rounding.py": '''"""Currency precision and rounding policy for billing."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

CURRENCY_QUANTUM: dict[str, Decimal] = {
    "USD": Decimal("0.01"),
    "EUR": Decimal("0.01"),
    "JPY": Decimal("1"),
}

# All invoice tax and total lines must round using ROUND_HALF_UP.
INVOICE_ROUNDING_MODE = ROUND_HALF_UP


def round_money(amount: Decimal, currency: str = "USD") -> Decimal:
    """Quantize a monetary amount using the package rounding policy."""
    quantum = CURRENCY_QUANTUM.get(currency.upper(), Decimal("0.01"))
    return amount.quantize(quantum, rounding=INVOICE_ROUNDING_MODE)
''',
            f"{pkg}/catalog.py": '''"""Product catalog metadata and tax-exemption rules."""

from __future__ import annotations

# Categories that are completely exempt from regional sales tax.
TAX_EXEMPT_CATEGORIES: frozenset[str] = frozenset(
    {"groceries", "medical", "education"}
)

# Maximum per-line promotional discount fraction allowed by finance policy.
MAX_DISCOUNT_RATE = 0.50


def is_tax_exempt(category: str) -> bool:
    """Return True when items in *category* must not be taxed."""
    return category.strip().lower() in TAX_EXEMPT_CATEGORIES
''',
            f"{pkg}/engine.py": '''"""Invoice total calculation engine."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from billing.rates import get_tax_rate


@dataclass(frozen=True)
class LineItem:
    sku: str
    category: str
    unit_price: Decimal
    quantity: int
    discount_rate: Decimal = Decimal("0")


def calculate_invoice_total(
    items: list[LineItem],
    region: str,
    currency: str = "USD",
) -> dict[str, Decimal]:
    """Return subtotal, tax, and total for an invoice."""
    subtotal = Decimal("0")
    taxable_subtotal = Decimal("0")
    for item in items:
        line_gross = item.unit_price * item.quantity
        # BUG 1: ignores discount_rate cap and does not subtract discount before tax
        # BUG 2: ignores tax-exempt categories
        subtotal += line_gross
        taxable_subtotal += line_gross
    tax_rate = get_tax_rate(region)
    tax = taxable_subtotal * tax_rate
    # BUG 3: returns unrounded floats/Decimals instead of using rounding policy
    total = subtotal + tax
    return {"subtotal": subtotal, "tax": tax, "total": total}
''',
            "tests/test_billing.py": '''"""Tests for invoice calculation across tax, exemption, and rounding rules."""

from decimal import Decimal

from billing.engine import LineItem, calculate_invoice_total


def test_standard_taxable_item_rounds_half_up() -> None:
    items = [LineItem("SKU-1", "electronics", Decimal("10.00"), 1)]
    res = calculate_invoice_total(items, region="US-NY", currency="USD")
    # 10.00 * 0.08875 = 0.8875 -> 0.89 with ROUND_HALF_UP
    assert res["subtotal"] == Decimal("10.00")
    assert res["tax"] == Decimal("0.89")
    assert res["total"] == Decimal("10.89")


def test_tax_exempt_category_has_zero_tax() -> None:
    items = [
        LineItem("SKU-2", "groceries", Decimal("20.00"), 2),
        LineItem("SKU-3", "Medical", Decimal("15.00"), 1),
    ]
    res = calculate_invoice_total(items, region="EU-DE", currency="EUR")
    assert res["subtotal"] == Decimal("55.00")
    assert res["tax"] == Decimal("0.00")
    assert res["total"] == Decimal("55.00")


def test_discount_applied_before_tax_and_capped_at_policy_max() -> None:
    items = [
        # 80% requested, must be capped at catalog MAX_DISCOUNT_RATE (50%)
        LineItem("SKU-4", "electronics", Decimal("100.00"), 1, Decimal("0.80")),
        # 20% discount on exempt education item
        LineItem("SKU-5", "education", Decimal("50.00"), 1, Decimal("0.20")),
    ]
    res = calculate_invoice_total(items, region="CA-ON", currency="USD")
    # 50.00 taxable + 40.00 exempt = 90.00 subtotal; tax = 50.00 * 0.13 = 6.50
    assert res["subtotal"] == Decimal("90.00")
    assert res["tax"] == Decimal("6.50")
    assert res["total"] == Decimal("96.50")


def test_jpy_uses_integer_quantum() -> None:
    items = [LineItem("SKU-6", "electronics", Decimal("1005"), 1)]
    res = calculate_invoice_total(items, region="US-CA", currency="JPY")
    # 1005 * 0.0825 = 82.9125 -> 83
    assert res["subtotal"] == Decimal("1005")
    assert res["tax"] == Decimal("83")
    assert res["total"] == Decimal("1088")
''',
        }
    )
    fixed_engine = '''"""Invoice total calculation engine."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from billing.catalog import MAX_DISCOUNT_RATE, is_tax_exempt
from billing.rates import get_tax_rate
from billing.rounding import round_money


@dataclass(frozen=True)
class LineItem:
    sku: str
    category: str
    unit_price: Decimal
    quantity: int
    discount_rate: Decimal = Decimal("0")


def calculate_invoice_total(
    items: list[LineItem],
    region: str,
    currency: str = "USD",
) -> dict[str, Decimal]:
    """Return subtotal, tax, and total for an invoice."""
    max_discount = Decimal(str(MAX_DISCOUNT_RATE))
    subtotal = Decimal("0")
    taxable_subtotal = Decimal("0")
    for item in items:
        effective_discount = min(max(item.discount_rate, Decimal("0")), max_discount)
        line_net = item.unit_price * item.quantity * (Decimal("1") - effective_discount)
        subtotal += line_net
        if not is_tax_exempt(item.category):
            taxable_subtotal += line_net
    tax_rate = get_tax_rate(region)
    rounded_subtotal = round_money(subtotal, currency)
    rounded_tax = round_money(taxable_subtotal * tax_rate, currency)
    rounded_total = round_money(rounded_subtotal + rounded_tax, currency)
    return {"subtotal": rounded_subtotal, "tax": rounded_tax, "total": rounded_total}
'''
    return TaskSpec(
        slug="invoice_tax_rounding",
        task_id="mem__invoice_tax_rounding",
        difficulty="medium",
        issue_text=(
            "Invoice totals are incorrect when orders include discounts, tax-exempt "
            "categories, or currencies with custom rounding precision.\n\n"
            "Investigate the codebase and fix the invoice total calculation so that:\n"
            "1. Per-line discounts are capped at the finance policy maximum and subtracted "
            "before computing subtotal and tax.\n"
            "2. Tax-exempt product categories are excluded from taxable subtotal.\n"
            "3. Subtotal, tax, and total are quantized according to the package currency "
            "and rounding policy.\n\n"
            "Ensure all tests pass."
        ),
        test_file="tests/test_billing.py",
        files=files,
        reference_overrides={f"{pkg}/engine.py": fixed_engine},
    )


def _build_rbac_task() -> TaskSpec:
    pkg = "authz"
    files = _common_repo_files(pkg, "rbac authorization")
    files.update(
        {
            f"{pkg}/roles.py": '''"""Role hierarchy and base permissions."""

from __future__ import annotations

# Maps role -> directly granted scopes.
ROLE_SCOPES: dict[str, frozenset[str]] = {
    "viewer": frozenset({"docs:read"}),
    "editor": frozenset({"docs:write"}),
    "auditor": frozenset({"audit:read"}),
    "admin": frozenset({"users:manage"}),
}

# Maps role -> parent roles whose scopes are inherited.
ROLE_PARENTS: dict[str, tuple[str, ...]] = {
    "viewer": (),
    "editor": ("viewer",),
    "auditor": ("viewer",),
    "admin": ("editor", "auditor"),
}


def expand_role_scopes(role: str) -> set[str]:
    """Return all scopes granted directly or transitively by *role*."""
    visited: set[str] = set()
    scopes: set[str] = set()
    stack = [role]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        scopes.update(ROLE_SCOPES.get(current, frozenset()))
        stack.extend(ROLE_PARENTS.get(current, ()))
    return scopes
''',
            f"{pkg}/scopes.py": '''"""Legacy scope aliases and canonical scope normalization."""

from __future__ import annotations

SCOPE_ALIASES: dict[str, str] = {
    "read:docs": "docs:read",
    "write:docs": "docs:write",
    "read:audit": "audit:read",
    "manage:users": "users:manage",
}


def normalize_scope(scope: str) -> str:
    """Map legacy scope names to canonical resource:action form."""
    cleaned = scope.strip().lower()
    return SCOPE_ALIASES.get(cleaned, cleaned)
''',
            f"{pkg}/tokens.py": '''"""Access token representation and expiration policy."""

from __future__ import annotations

from dataclasses import dataclass

# Tokens remain acceptable for up to 30 seconds past exp to tolerate clock skew.
EXPIRY_GRACE_SECONDS = 30


@dataclass(frozen=True)
class AccessToken:
    subject: str
    roles: tuple[str, ...]
    direct_scopes: tuple[str, ...]
    expires_at: int
    revoked: bool = False


def is_token_active(token: AccessToken, now: int) -> bool:
    """Return True if *token* is not revoked and within expiry + grace window."""
    if token.revoked:
        return False
    return now <= token.expires_at + EXPIRY_GRACE_SECONDS
''',
            f"{pkg}/guard.py": '''"""Endpoint authorization guard."""

from __future__ import annotations

from authz.roles import ROLE_SCOPES
from authz.tokens import AccessToken


def authorize(token: AccessToken, required_scope: str, now: int) -> bool:
    """Return True if *token* is allowed to perform *required_scope* at *now*."""
    # BUG 1: ignores token revocation and expiry grace window
    if now > token.expires_at:
        return False
    # BUG 2: does not normalize legacy scope aliases
    # BUG 3: checks only direct ROLE_SCOPES without inherited parent roles
    granted: set[str] = set(token.direct_scopes)
    for role in token.roles:
        granted.update(ROLE_SCOPES.get(role, frozenset()))
    return required_scope in granted
''',
            "tests/test_authz.py": '''"""Tests for RBAC guard across hierarchy, aliases, and token grace window."""

from authz.guard import authorize
from authz.tokens import AccessToken


def test_admin_inherits_viewer_and_auditor_scopes() -> None:
    tok = AccessToken("alice", roles=("admin",), direct_scopes=(), expires_at=1000)
    assert authorize(tok, "docs:read", now=900) is True
    assert authorize(tok, "audit:read", now=900) is True
    assert authorize(tok, "users:manage", now=900) is True


def test_legacy_scope_aliases_accepted_on_request_and_token() -> None:
    tok = AccessToken("bob", roles=(), direct_scopes=("write:docs",), expires_at=1000)
    assert authorize(tok, "docs:write", now=950) is True
    assert authorize(tok, "write:docs", now=950) is True
    assert authorize(tok, "docs:read", now=950) is False


def test_expiry_grace_window_and_revocation() -> None:
    tok = AccessToken("carol", roles=("editor",), direct_scopes=(), expires_at=1000)
    # Within 30s grace window
    assert authorize(tok, "docs:read", now=1025) is True
    # Beyond 30s grace window
    assert authorize(tok, "docs:read", now=1031) is False
    revoked = AccessToken(
        "carol", roles=("editor",), direct_scopes=(), expires_at=2000, revoked=True
    )
    assert authorize(revoked, "docs:read", now=1000) is False
''',
        }
    )
    fixed_guard = '''"""Endpoint authorization guard."""

from __future__ import annotations

from authz.roles import expand_role_scopes
from authz.scopes import normalize_scope
from authz.tokens import AccessToken, is_token_active


def authorize(token: AccessToken, required_scope: str, now: int) -> bool:
    """Return True if *token* is allowed to perform *required_scope* at *now*."""
    if not is_token_active(token, now):
        return False
    target = normalize_scope(required_scope)
    granted: set[str] = {normalize_scope(s) for s in token.direct_scopes}
    for role in token.roles:
        for scope in expand_role_scopes(role):
            granted.add(normalize_scope(scope))
    return target in granted
'''
    return TaskSpec(
        slug="rbac_token_scope",
        task_id="mem__rbac_token_scope",
        difficulty="medium",
        issue_text=(
            "Authorization checks are failing for delegated roles, legacy scope names, "
            "and tokens near expiration.\n\n"
            "Investigate the authorization package and fix the endpoint guard so that:\n"
            "1. Revoked tokens are always rejected, while non-revoked tokens honor the "
            "configured expiration grace window.\n"
            "2. Both direct token scopes and requested scopes normalize legacy aliases.\n"
            "3. Role permissions include all transitively inherited parent roles.\n\n"
            "Ensure all tests pass."
        ),
        test_file="tests/test_authz.py",
        files=files,
        reference_overrides={f"{pkg}/guard.py": fixed_guard},
    )


def _build_cache_task() -> TaskSpec:
    pkg = "cachelib"
    files = _common_repo_files(pkg, "cache eviction")
    files.update(
        {
            f"{pkg}/namespaces.py": '''"""Key namespace conventions and eviction pinning rules."""

from __future__ import annotations

# Keys with these prefixes are pinned and must not be evicted before unpinned keys.
PINNED_PREFIXES: tuple[str, ...] = ("cfg:", "auth:")


def extract_namespace(key: str) -> str:
    """Return the prefix up to and including ':' or '' if none."""
    if ":" not in key:
        return ""
    return key.split(":", 1)[0] + ":"


def is_pinned_key(key: str) -> bool:
    """Return True when *key* belongs to a pinned namespace."""
    return extract_namespace(key) in PINNED_PREFIXES
''',
            f"{pkg}/policies.py": '''"""Per-namespace TTL overrides."""

from __future__ import annotations

DEFAULT_TTL_SECONDS = 60

NAMESPACE_TTL_OVERRIDES: dict[str, int] = {
    "cfg:": 300,
    "auth:": 120,
    "tmp:": 10,
}


def resolve_ttl(namespace: str, explicit_ttl: int | None) -> int:
    """Use explicit_ttl if given, else namespace override, else default TTL."""
    if explicit_ttl is not None:
        return explicit_ttl
    return NAMESPACE_TTL_OVERRIDES.get(namespace, DEFAULT_TTL_SECONDS)
''',
            f"{pkg}/clock.py": '''"""Clock helper with skew tolerance for cache expiration."""

from __future__ import annotations

# Entries are considered expired once now >= expires_at + SKEW_TOLERANCE_SECONDS.
SKEW_TOLERANCE_SECONDS = 2


def is_expired(expires_at: int, now: int) -> bool:
    """Return True when an entry is expired accounting for clock skew."""
    return now >= expires_at + SKEW_TOLERANCE_SECONDS
''',
            f"{pkg}/lru.py": '''"""Bounded in-memory LRU cache store."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from cachelib.policies import DEFAULT_TTL_SECONDS


@dataclass
class _Entry:
    value: object
    expires_at: int


class BoundedCache:
    """Fixed-capacity LRU cache with TTL expiration."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, _Entry] = OrderedDict()

    def set(self, key: str, value: object, now: int, ttl: int | None = None) -> None:
        # BUG 1: ignores namespace TTL overrides
        effective_ttl = ttl if ttl is not None else DEFAULT_TTL_SECONDS
        if key in self._items:
            self._items.move_to_end(key)
        self._items[key] = _Entry(value=value, expires_at=now + effective_ttl)
        # BUG 2: evicts oldest key even if it is pinned while unpinned keys exist
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def get(self, key: str, now: int) -> object | None:
        entry = self._items.get(key)
        if entry is None:
            return None
        # BUG 3: does not account for clock skew tolerance
        if now >= entry.expires_at:
            del self._items[key]
            return None
        self._items.move_to_end(key)
        return entry.value
''',
            "tests/test_cache.py": '''"""Tests for bounded LRU cache across namespaces, TTLs, and clock skew."""

from cachelib.lru import BoundedCache


def test_namespace_ttl_override_and_clock_skew() -> None:
    cache = BoundedCache(capacity=4)
    cache.set("tmp:session", "v1", now=100)  # tmp: TTL is 10 -> expires_at=110
    # At now=111, within 2s skew tolerance -> still valid
    assert cache.get("tmp:session", now=111) == "v1"
    # At now=112 (>= 110 + 2) -> expired
    assert cache.get("tmp:session", now=112) is None


def test_pinned_prefix_survives_eviction_over_unpinned() -> None:
    cache = BoundedCache(capacity=2)
    cache.set("cfg:theme", "dark", now=0)  # pinned, oldest
    cache.set("user:1", "alice", now=1)  # unpinned, newer
    cache.set("user:2", "bob", now=2)  # triggers eviction -> must evict user:1, not cfg:theme
    assert cache.get("cfg:theme", now=3) == "dark"
    assert cache.get("user:1", now=3) is None
    assert cache.get("user:2", now=3) == "bob"


def test_expired_entries_purged_before_evicting_live_entries() -> None:
    cache = BoundedCache(capacity=2)
    cache.set("tmp:old", "x", now=0)  # expires_at=10, expired at 12
    cache.set("user:live", "y", now=5)
    cache.set("user:new", "z", now=20)
    assert cache.get("tmp:old", now=20) is None
    assert cache.get("user:live", now=20) == "y"
    assert cache.get("user:new", now=20) == "z"
''',
        }
    )
    fixed_lru = '''"""Bounded in-memory LRU cache store."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from cachelib.clock import is_expired
from cachelib.namespaces import extract_namespace, is_pinned_key
from cachelib.policies import resolve_ttl


@dataclass
class _Entry:
    value: object
    expires_at: int


class BoundedCache:
    """Fixed-capacity LRU cache with TTL expiration."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, _Entry] = OrderedDict()

    def _purge_expired(self, now: int) -> None:
        expired_keys = [
            k for k, entry in self._items.items() if is_expired(entry.expires_at, now)
        ]
        for k in expired_keys:
            del self._items[k]

    def set(self, key: str, value: object, now: int, ttl: int | None = None) -> None:
        self._purge_expired(now)
        namespace = extract_namespace(key)
        effective_ttl = resolve_ttl(namespace, ttl)
        if key in self._items:
            self._items.move_to_end(key)
        self._items[key] = _Entry(value=value, expires_at=now + effective_ttl)
        while len(self._items) > self.capacity:
            victim = next(
                (k for k in self._items if not is_pinned_key(k)),
                next(iter(self._items)),
            )
            del self._items[victim]

    def get(self, key: str, now: int) -> object | None:
        entry = self._items.get(key)
        if entry is None:
            return None
        if is_expired(entry.expires_at, now):
            del self._items[key]
            return None
        self._items.move_to_end(key)
        return entry.value
'''
    return TaskSpec(
        slug="cache_eviction_ttl",
        task_id="mem__cache_eviction_ttl",
        difficulty="medium",
        issue_text=(
            "The in-memory cache is evicting pinned configuration keys prematurely and "
            "expiring namespace-prefixed keys at the wrong time.\n\n"
            "Investigate the cache package and fix the cache store so that:\n"
            "1. When no explicit TTL is passed, namespace TTL overrides are applied.\n"
            "2. Expiration checks respect the configured clock skew tolerance, and "
            "expired entries are purged before evicting live items.\n"
            "3. Capacity eviction evicts the oldest unpinned key before evicting any "
            "pinned namespace key.\n\n"
            "Ensure all tests pass."
        ),
        test_file="tests/test_cache.py",
        files=files,
        reference_overrides={f"{pkg}/lru.py": fixed_lru},
    )


def _build_retry_task() -> TaskSpec:
    pkg = "jobrunner"
    files = _common_repo_files(pkg, "job retry scheduler")
    files.update(
        {
            f"{pkg}/errors.py": '''"""Job error classification."""

from __future__ import annotations

FATAL_ERROR_CODES: frozenset[str] = frozenset(
    {"INVALID_PAYLOAD", "AUTH_FAILED", "SCHEMA_MISMATCH"}
)

RETRYABLE_ERROR_CODES: frozenset[str] = frozenset(
    {"TIMEOUT", "RATE_LIMITED", "CONNECTION_RESET"}
)


def is_retryable(error_code: str) -> bool:
    """Return True only for explicitly retryable, non-fatal error codes."""
    code = error_code.strip().upper()
    if code in FATAL_ERROR_CODES:
        return False
    return code in RETRYABLE_ERROR_CODES
''',
            f"{pkg}/queues.py": '''"""Per-queue retry configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueueConfig:
    base_delay_seconds: int
    backoff_multiplier: int
    max_attempts: int


QUEUE_CONFIGS: dict[str, QueueConfig] = {
    "critical": QueueConfig(base_delay_seconds=2, backoff_multiplier=2, max_attempts=5),
    "default": QueueConfig(base_delay_seconds=5, backoff_multiplier=2, max_attempts=3),
    "bulk": QueueConfig(base_delay_seconds=10, backoff_multiplier=3, max_attempts=4),
}


def get_queue_config(queue_name: str) -> QueueConfig:
    return QUEUE_CONFIGS.get(queue_name, QUEUE_CONFIGS["default"])
''',
            f"{pkg}/limits.py": '''"""Global backoff safety caps."""

from __future__ import annotations

# No single retry delay may exceed this global ceiling.
MAX_BACKOFF_SECONDS = 60


def clamp_delay(seconds: int) -> int:
    """Clamp a computed retry delay to [0, MAX_BACKOFF_SECONDS]."""
    return max(0, min(seconds, MAX_BACKOFF_SECONDS))
''',
            f"{pkg}/scheduler.py": '''"""Retry scheduling decisions for failed jobs."""

from __future__ import annotations

from dataclasses import dataclass

from jobrunner.queues import get_queue_config


@dataclass(frozen=True)
class RetryDecision:
    should_retry: bool
    delay_seconds: int
    reason: str


def plan_retry(queue_name: str, error_code: str, attempt: int) -> RetryDecision:
    """Decide whether a failed job on *attempt* (1-based) should retry and after how long."""
    cfg = get_queue_config(queue_name)
    # BUG 1: ignores error classification (retries fatal errors)
    # BUG 2: uses <= instead of < so it allows one extra retry beyond max_attempts
    if attempt > cfg.max_attempts:
        return RetryDecision(False, 0, "max_attempts_reached")
    # BUG 3: does not clamp exponential delay to global MAX_BACKOFF_SECONDS
    delay = cfg.base_delay_seconds * (cfg.backoff_multiplier ** (attempt - 1))
    return RetryDecision(True, delay, "retry_scheduled")
''',
            "tests/test_scheduler.py": '''"""Tests for job retry planning across error codes, queues, and delay caps."""

from jobrunner.scheduler import plan_retry


def test_fatal_and_unknown_errors_never_retry() -> None:
    res = plan_retry("critical", "INVALID_PAYLOAD", attempt=1)
    assert res.should_retry is False
    assert res.delay_seconds == 0

    unknown = plan_retry("default", "MYSTERY_BUG", attempt=1)
    assert unknown.should_retry is False


def test_retryable_error_honors_max_attempts_boundary() -> None:
    # default queue has max_attempts=3 -> attempt 1 and 2 can retry, attempt 3 stops
    assert plan_retry("default", "TIMEOUT", attempt=1).should_retry is True
    assert plan_retry("default", "TIMEOUT", attempt=2).should_retry is True
    stop = plan_retry("default", "TIMEOUT", attempt=3)
    assert stop.should_retry is False
    assert stop.delay_seconds == 0


def test_exponential_backoff_clamped_to_global_cap() -> None:
    # bulk queue: base=10, mult=3, max_attempts=4
    # attempt 1 -> 10, attempt 2 -> 30, attempt 3 -> 90 clamped to 60
    assert plan_retry("bulk", "RATE_LIMITED", attempt=1).delay_seconds == 10
    assert plan_retry("bulk", "RATE_LIMITED", attempt=2).delay_seconds == 30
    capped = plan_retry("bulk", "RATE_LIMITED", attempt=3)
    assert capped.should_retry is True
    assert capped.delay_seconds == 60
''',
        }
    )
    fixed_scheduler = '''"""Retry scheduling decisions for failed jobs."""

from __future__ import annotations

from dataclasses import dataclass

from jobrunner.errors import is_retryable
from jobrunner.limits import clamp_delay
from jobrunner.queues import get_queue_config


@dataclass(frozen=True)
class RetryDecision:
    should_retry: bool
    delay_seconds: int
    reason: str


def plan_retry(queue_name: str, error_code: str, attempt: int) -> RetryDecision:
    """Decide whether a failed job on *attempt* (1-based) should retry and after how long."""
    if not is_retryable(error_code):
        return RetryDecision(False, 0, "non_retryable_error")
    cfg = get_queue_config(queue_name)
    if attempt >= cfg.max_attempts:
        return RetryDecision(False, 0, "max_attempts_reached")
    raw_delay = cfg.base_delay_seconds * (cfg.backoff_multiplier ** (attempt - 1))
    return RetryDecision(True, clamp_delay(raw_delay), "retry_scheduled")
'''
    return TaskSpec(
        slug="pipeline_retry_backoff",
        task_id="mem__pipeline_retry_backoff",
        difficulty="medium",
        issue_text=(
            "The job retry scheduler is retrying fatal errors, scheduling one more "
            "attempt than each queue allows, and exceeding the global backoff ceiling.\n\n"
            "Investigate the job runner package and fix retry planning so that:\n"
            "1. Only explicitly retryable (non-fatal) error codes are retried.\n"
            "2. Once a job reaches the queue's configured max_attempts, no further retry "
            "is scheduled.\n"
            "3. Exponential backoff delays are clamped to the global maximum delay.\n\n"
            "Ensure all tests pass."
        ),
        test_file="tests/test_scheduler.py",
        files=files,
        reference_overrides={f"{pkg}/scheduler.py": fixed_scheduler},
    )


def _build_shipping_task() -> TaskSpec:
    pkg = "shipping"
    files = _common_repo_files(pkg, "shipping calculator")
    files.update(
        {
            f"{pkg}/zones.py": '''"""Shipping zone multipliers and base rates."""

from __future__ import annotations

from decimal import Decimal

BASE_PER_KG = Decimal("3.50")

ZONE_MULTIPLIERS: dict[str, Decimal] = {
    "LOCAL": Decimal("1.0"),
    "REGIONAL": Decimal("1.4"),
    "NATIONAL": Decimal("1.8"),
    "INTL": Decimal("2.5"),
}


def get_zone_multiplier(zone: str) -> Decimal:
    return ZONE_MULTIPLIERS[zone.strip().upper()]
''',
            f"{pkg}/surcharges.py": '''"""Special handling surcharges."""

from __future__ import annotations

from decimal import Decimal

FRAGILE_SURCHARGE = Decimal("7.50")
HEAVY_WEIGHT_THRESHOLD_KG = Decimal("10.0")
HEAVY_SURCHARGE = Decimal("12.00")


def compute_surcharges(weight_kg: Decimal, fragile: bool) -> Decimal:
    """Return total handling surcharges (never waived by free-shipping promos)."""
    total = Decimal("0.00")
    if fragile:
        total += FRAGILE_SURCHARGE
    if weight_kg > HEAVY_WEIGHT_THRESHOLD_KG:
        total += HEAVY_SURCHARGE
    return total
''',
            f"{pkg}/promotions.py": '''"""Free-shipping promotion eligibility rules."""

from __future__ import annotations

from decimal import Decimal

FREE_SHIPPING_MIN_ORDER = Decimal("100.00")
ELIGIBLE_PROMO_ZONES: frozenset[str] = frozenset({"LOCAL", "REGIONAL", "NATIONAL"})


def qualifies_for_free_base_shipping(order_subtotal: Decimal, zone: str) -> bool:
    """Return True when base weight shipping is waived (handling surcharges still apply)."""
    return (
        order_subtotal >= FREE_SHIPPING_MIN_ORDER
        and zone.strip().upper() in ELIGIBLE_PROMO_ZONES
    )
''',
            f"{pkg}/calculator.py": '''"""Final shipping quote calculator."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from shipping.zones import BASE_PER_KG, get_zone_multiplier


def quote_shipping(
    weight_kg: Decimal,
    zone: str,
    order_subtotal: Decimal,
    fragile: bool = False,
) -> Decimal:
    """Return the final quantized USD shipping cost."""
    # BUG 1: ignores free-shipping promotion for eligible zones/subtotals
    # BUG 2: ignores fragile and heavy-weight surcharges
    cost = weight_kg * BASE_PER_KG * get_zone_multiplier(zone)
    return cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
''',
            "tests/test_shipping.py": '''"""Tests for shipping quotes across zones, surcharges, and free-shipping promos."""

from decimal import Decimal

from shipping.calculator import quote_shipping


def test_standard_regional_shipment_without_promo() -> None:
    # 4 kg * 3.50 * 1.4 = 19.60
    assert quote_shipping(Decimal("4"), "REGIONAL", Decimal("50.00")) == Decimal("19.60")


def test_free_shipping_waives_base_cost_but_keeps_surcharges() -> None:
    # order >= 100 in NATIONAL -> base shipping waived, fragile (7.50) + heavy (12.00) remain
    cost = quote_shipping(
        Decimal("12"), "NATIONAL", Decimal("150.00"), fragile=True
    )
    assert cost == Decimal("19.50")


def test_international_never_qualifies_for_free_shipping() -> None:
    # 2 kg * 3.50 * 2.5 = 17.50 + fragile 7.50 = 25.00
    cost = quote_shipping(Decimal("2"), "INTL", Decimal("500.00"), fragile=True)
    assert cost == Decimal("25.00")


def test_exact_heavy_threshold_not_charged_heavy_fee() -> None:
    # 10.0 kg is NOT > 10.0 kg -> no heavy fee; LOCAL base = 10 * 3.50 * 1.0 = 35.00
    assert quote_shipping(Decimal("10.0"), "LOCAL", Decimal("40.00")) == Decimal("35.00")
''',
        }
    )
    fixed_calc = '''"""Final shipping quote calculator."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from shipping.promotions import qualifies_for_free_base_shipping
from shipping.surcharges import compute_surcharges
from shipping.zones import BASE_PER_KG, get_zone_multiplier


def quote_shipping(
    weight_kg: Decimal,
    zone: str,
    order_subtotal: Decimal,
    fragile: bool = False,
) -> Decimal:
    """Return the final quantized USD shipping cost."""
    if qualifies_for_free_base_shipping(order_subtotal, zone):
        base_cost = Decimal("0.00")
    else:
        base_cost = weight_kg * BASE_PER_KG * get_zone_multiplier(zone)
    total = base_cost + compute_surcharges(weight_kg, fragile)
    return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
'''
    return TaskSpec(
        slug="shipping_zone_rates",
        task_id="mem__shipping_zone_rates",
        difficulty="medium",
        issue_text=(
            "Shipping quotes are wrong for orders that qualify for free shipping, "
            "international destinations, or fragile/heavy parcels.\n\n"
            "Investigate the shipping package and fix shipping quote calculation so that:\n"
            "1. Orders meeting the free-shipping promotion rules have base weight shipping "
            "waived (only in eligible zones).\n"
            "2. Fragile and heavy-weight handling surcharges are always applied and never "
            "waived by free shipping.\n"
            "3. Final cost is quantized to 2 decimal places.\n\n"
            "Ensure all tests pass."
        ),
        test_file="tests/test_shipping.py",
        files=files,
        reference_overrides={f"{pkg}/calculator.py": fixed_calc},
    )


ALL_TASKS: tuple[TaskSpec, ...] = (
    _build_invoice_task(),
    _build_rbac_task(),
    _build_cache_task(),
    _build_retry_task(),
    _build_shipping_task(),
)


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _write_files(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


def build_suite(repos_root: Path, suite_dir: Path, *, verify: bool = False) -> None:
    repos_root.mkdir(parents=True, exist_ok=True)
    suite_dir.mkdir(parents=True, exist_ok=True)

    for spec in ALL_TASKS:
        repo_dir = (repos_root / spec.slug).resolve()
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.mkdir(parents=True)
        _write_files(repo_dir, spec.files)
        _git(repo_dir, "init", "-b", "main")
        _git(repo_dir, "config", "user.email", "benchmark@local")
        _git(repo_dir, "config", "user.name", "Benchmark Builder")
        _git(repo_dir, "add", ".")
        _git(repo_dir, "commit", "-m", f"Initial buggy {spec.slug} repo")
        base_commit = _git(repo_dir, "rev-parse", "HEAD")

        if verify:
            fail_run = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", spec.test_file],
                cwd=repo_dir,
                capture_output=True,
                text=True,
            )
            if fail_run.returncode == 0:
                raise RuntimeError(
                    f"{spec.task_id} unexpectedly passed at base_commit!"
                )
            with tempfile.TemporaryDirectory() as tmp:
                tmp_repo = Path(tmp) / spec.slug
                shutil.copytree(repo_dir, tmp_repo)
                _write_files(tmp_repo, spec.reference_overrides)
                pass_run = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", spec.test_file],
                    cwd=tmp_repo,
                    capture_output=True,
                    text=True,
                )
                if pass_run.returncode != 0:
                    raise RuntimeError(
                        f"{spec.task_id} failed with reference fix:\n{pass_run.stdout}\n{pass_run.stderr}"
                    )

        manifest = {
            "task_id": spec.task_id,
            "repo": str(repo_dir),
            "base_commit": base_commit,
            "issue_text": spec.issue_text,
            "oracle_tests": [spec.test_file],
            "oracle_tier": "local_full",
            "agent_test_targets": [spec.test_file],
            "metadata": {
                "source": "local/memory_lookup",
                "slug": spec.slug,
                "difficulty": spec.difficulty,
                "test_file": spec.test_file,
                "file_count": len(spec.files),
            },
        }
        (suite_dir / f"{spec.slug}.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[BUILT] {spec.task_id:28s} files={len(spec.files)} commit={base_commit[:10]}"
        )

    loaded = load_suite_dir(suite_dir)
    print(f"Verified load_suite_dir({suite_dir}): {len(loaded)} manifests loaded.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repos-root", type=Path, default=Path(".runs/memory_lookup_repos")
    )
    parser.add_argument(
        "--suite-dir", type=Path, default=Path("benchmarks/memory_lookup")
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    build_suite(args.repos_root, args.suite_dir, verify=args.verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
