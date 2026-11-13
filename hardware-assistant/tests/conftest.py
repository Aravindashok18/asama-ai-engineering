import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hwqa.system import FixtureSystem  # noqa: E402

FIXTURES = ROOT / "fixtures"


@pytest.fixture
def thinkpad():
    return FixtureSystem(str(FIXTURES / "thinkpad-e14"), sudo_allowed=True)


@pytest.fixture
def thinkpad_no_root():
    return FixtureSystem(str(FIXTURES / "thinkpad-e14"), sudo_allowed=False)


@pytest.fixture
def legion():
    return FixtureSystem(str(FIXTURES / "legion-5"))


@pytest.fixture
def vm():
    return FixtureSystem(str(FIXTURES / "minimal-vm"), sudo_allowed=True)
