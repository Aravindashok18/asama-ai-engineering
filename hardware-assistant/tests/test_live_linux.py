"""Runs the real tools against the machine running the tests. Skipped off Linux.

Run it on a Linux machine to check the live command execution and parsing path, not only
the recorded fixtures.
"""
import json
import platform

import pytest

from hwqa.assistant import Assistant
from hwqa.system import LiveSystem

pytestmark = pytest.mark.skipif(platform.system() != "Linux", reason="needs a Linux host")


@pytest.fixture(scope="module")
def dump():
    return Assistant(LiveSystem()).collect_all()


def test_every_tool_returns_without_error(dump):
    for name, result in dump.items():
        assert "error" not in result, (name, result)


def test_core_facts_are_read(dump):
    assert dump["get_cpu_info"]["data"]["threads"] >= 1
    assert dump["get_memory_info"]["data"]["usable_gb"] > 0
    assert dump["get_storage_info"]["data"]["disks"]
    assert dump["get_disk_usage"]["data"]["filesystems"]
    assert dump["get_system_info"]["data"]["kernel"]


def test_no_root_means_gap_not_crash(dump):
    mem = dump["get_memory_info"]
    if "upgrade" not in mem["data"]:
        assert mem["unavailable"], "missing slot data must be explained"


def test_rule_mode_end_to_end():
    ans = Assistant(LiveSystem()).ask("How much RAM do I have?")
    assert "GB" in ans.text and ans.unverified == []


def test_live_output_has_no_mac_addresses(dump):
    import re

    assert not re.search(r"\b[0-9a-f]{2}(:[0-9a-f]{2}){5}\b", json.dumps(dump), re.I)
