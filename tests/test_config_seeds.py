"""Tests for replication seed resolution."""

from __future__ import annotations

import pytest

from src.config import resolve_replication_seeds


def test_resolve_replication_seeds_from_cli() -> None:
    assert resolve_replication_seeds(cli_seeds="0,2,4", run_config={"num_seeds": 3}) == [0, 2, 4]


def test_resolve_replication_seeds_from_yaml_list() -> None:
    assert resolve_replication_seeds(run_config={"seeds": [1, 3, 5]}) == [1, 3, 5]


def test_resolve_replication_seeds_from_num_seeds() -> None:
    assert resolve_replication_seeds(run_config={"num_seeds": 10}) == list(range(10))


def test_resolve_replication_seeds_default_count() -> None:
    assert resolve_replication_seeds(run_config={}) == list(range(10))


def test_resolve_replication_seeds_rejects_empty_cli() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        resolve_replication_seeds(cli_seeds=",")
