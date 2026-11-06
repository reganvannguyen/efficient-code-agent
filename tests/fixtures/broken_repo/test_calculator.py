"""Targeted tests for calculator module."""

from calculator import calculate_total


def test_calculate_total_adds_tax() -> None:
    """Subtotal $100 with 5% tax should equal $105.00."""
    assert calculate_total(100.0, 0.05) == 105.0

