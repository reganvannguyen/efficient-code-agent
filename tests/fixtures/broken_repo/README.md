# Calculator fixture with defect

This repository contains a deliberately broken `calculate_total` function.
`calculate_total(subtotal, tax_rate)` erroneously subtracts the tax instead of
adding it. `test_calculator.py` asserts the correct addition behavior.

