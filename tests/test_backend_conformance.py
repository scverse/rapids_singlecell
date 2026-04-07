"""Run squidpy's backend conformance suite against the RSC backend."""

from __future__ import annotations

from squidpy.testing.backend_conformance import validate_backend


def test_conformance():
    results = validate_backend("rapids_singlecell")
    for name, status in results.items():
        assert status == "PASSED", f"{name}: {status}"
