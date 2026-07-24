"""Phase 0 smoke test: package cài được và import được."""

import tripkit


def test_import_tripkit():
    assert tripkit.__version__
