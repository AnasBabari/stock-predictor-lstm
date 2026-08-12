"""Backend test package boundary.

The package marker keeps helper imports such as ``tests.test_api`` stable and
prevents same-named modules in the backend and research suites from colliding
when both suites are collected together.
"""
