"""Financial calculations helper."""


def calculate_total(subtotal: float, tax_rate: float) -> float:
    """Calculate the total price including tax, rounded to 2 decimal places."""
    # Defect: subtracts tax instead of adding it
    return round(subtotal - (subtotal * tax_rate), 2)

