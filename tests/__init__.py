"""Test package.

Present so `tests` is a regular package rather than a PEP 420 namespace
package. The test modules import shared fixtures helpers via
`from tests.conftest import ...`, which otherwise depends on namespace
resolution that does not hold reliably across Python/pytest versions --
it resolved on 3.11 and failed on 3.13 with ModuleNotFoundError. Making
the package explicit also means conftest is imported once, as
`tests.conftest`, instead of once as top-level `conftest` and again as
`tests.conftest`.
"""
