"""Research test package boundary.

Keeping the research tests importable as a package prevents duplicate module
names (for example, the backend and research ``test_multi_seed.py`` files)
from colliding when both suites are collected in one pytest invocation.
"""
