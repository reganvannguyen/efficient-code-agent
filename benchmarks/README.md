# Benchmark Task Suites

## Dataset separation

- `benchmarks/smoke/` — **5-task smoke suite** for validating the baseline runner, artifact collection, and macOS resource measurement.
- `benchmarks/dev/` — Reserved for development comparisons and tuning (Phase 5/6).
- `benchmarks/held_out/` — Reserved for final untouched evaluation; **do not tune on held-out tasks**.

> **Note:** Local runs over these manifests use `oracle_tier = "local_targeted"` and are **not** official SWE-bench Verified scores.

## Selection criteria (macOS ARM, 16 GB MacBook Air)

1. Pure-Python implementation and test suite (no C/Cython/Fortran compilation).
2. No external network calls, paid APIs, or database servers required for the selected test target.
3. Clear issue description and isolated `pytest` target file.

## Included smoke tasks (5)

| Task ID | Repository | Base commit | Oracle tier | Oracle test target |
| --- | --- | --- | --- | --- |
| `pallets__flask-5014` | `pallets/flask` | `7ee9ceb71e868944a46e1ff00b506772a53a4f1d` | `local_targeted` | `tests/test_blueprints.py` |
| `pylint-dev__pylint-4551` | `pylint-dev/pylint` | `99589b08de8c5a2c6cc61e13a37420a868c80599` | `local_targeted` | `tests/unittest_pyreverse_writer.py` |
| `pytest-dev__pytest-10051` | `pytest-dev/pytest` | `aa55975c7d3f6c9f6d7f68accc41bb7cadf0eb9a` | `local_targeted` | `testing/logging/test_fixture.py` |
| `pytest-dev__pytest-10081` | `pytest-dev/pytest` | `da9a2b584eb7a6c7e924b2621ed0ddaeca0a7bea` | `local_targeted` | `testing/test_unittest.py` |
| `sympy__sympy-12096` | `sympy/sympy` | `d7c3045115693e887bcd03599b7ca4650ac5f2cb` | `local_targeted` | `sympy/utilities/tests/test_lambdify.py` |

## Excluded candidates and reasons

| Repository / Candidate | Reason for exclusion |
| --- | --- |
| `scikit-learn/scikit-learn`, `matplotlib/matplotlib`, `astropy/astropy` | Require C/Cython/Fortran native builds unsuitable for lightweight per-checkout smoke testing. |
| `pydata/xarray`, `mwaskom/seaborn` | Heavy scientific Python dependency stacks (`numpy`/`pandas`/`scipy`). |
| `django/django` | Large test harness and database/settings setup overhead for initial 5-task smoke validation. |
| `psf/requests` (`psf__requests-1724`, `psf__requests-2317`) | Test suite relies on external HTTP endpoints (`httpbin.org`) or legacy Python 2 encoding behavior. |
