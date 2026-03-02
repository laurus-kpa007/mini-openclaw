"""Shared test fixtures for mini-openclaw."""

import pytest

from mini_openclaw.config import AppConfig
from mini_openclaw.core.gateway import Gateway


@pytest.fixture
def config():
    """Default test configuration."""
    return AppConfig()


@pytest.fixture
def gateway(config):
    """Gateway instance with default config (not started)."""
    return Gateway(config)
