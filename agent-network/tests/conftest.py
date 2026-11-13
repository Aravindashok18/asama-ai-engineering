import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from agentnet.api import create_app  # noqa: E402
from agentnet.seed import in_memory_platform, seed_platform  # noqa: E402

ADMIN = {"X-Admin-Key": "dev-admin-key"}


@pytest.fixture
def platform():
    return in_memory_platform()


@pytest.fixture
def seeded(platform):
    return platform, seed_platform(platform)


@pytest.fixture
def api(platform):
    return TestClient(create_app(platform))


def auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}
