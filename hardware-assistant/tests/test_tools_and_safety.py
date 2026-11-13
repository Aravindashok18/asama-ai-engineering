import json
import subprocess

import pytest

from hwqa import system as S
from hwqa.tools import TOOLS, ToolError, call_tool, capability_report, redact, validate_args


def test_memory_upgrade_matches_the_brief(thinkpad):
    r = call_tool(thinkpad, "get_memory_info")
    up = r["data"]["upgrade"]
    assert r["data"]["installed_gb"] == 16
    assert (up["used_slots"], up["total_slots"], up["free_slots"]) == (1, 2, 1)
    assert up["max_supported_gb"] == 64 and up["max_per_slot_gb"] == 32
    assert up["max_total_gb_keeping_current_modules"] == 48
    assert "dmidecode -t 16 -t 17" in r["sources"]


def test_memory_without_root_reports_the_gap_not_a_guess(thinkpad_no_root):
    r = call_tool(thinkpad_no_root, "get_memory_info")
    assert "upgrade" not in r["data"] and "installed_gb" not in r["data"]
    assert r["data"]["usable_gb"] == 14.9
    gap = r["unavailable"][0]
    assert "needs root" in gap["reason"] and "sudoers" in gap["how_to_enable"]


def test_full_slots_and_running_as_root(legion):
    up = call_tool(legion, "get_memory_info")["data"]["upgrade"]
    assert up["free_slots"] == 0 and up["max_addable_gb_keeping_current_modules"] == 0


def test_storage_and_gpu(thinkpad, legion):
    disks = call_tool(thinkpad, "get_storage_info")["data"]["disks"]
    assert [d["kind"] for d in disks] == ["NVMe SSD"]  # zram filtered out
    assert disks[0]["size"] == "512 GB"
    gpus = call_tool(legion, "get_gpu_info")["data"]["gpus"]
    assert gpus[0]["vram_gb"] == 6.0 and gpus[0]["nvidia_driver"] == "545.29.06"
    assert gpus[1]["driver"] == "amdgpu"


def test_battery_health_computed(thinkpad):
    b = call_tool(thinkpad, "get_battery_info")["data"]["batteries"][0]
    assert b["health_percent"] == 87 and b["cycle_count"] == 212
    assert call_tool(thinkpad, "get_battery_info")["data"]["on_ac_power"] is False


def test_vm_fallbacks_and_missing_tools(vm):
    disks = call_tool(vm, "get_storage_info")["data"]["disks"]
    assert [d["device"] for d in disks] == ["/dev/vda"]  # /sys/block fallback, loop0 skipped
    gpu = call_tool(vm, "get_gpu_info")
    assert gpu["data"]["gpus"] == [] and "pciutils" in gpu["unavailable"][0]["how_to_enable"]
    caps = capability_report(vm)
    assert caps["get_memory_info"] == "partial: dmidecode not installed"
    assert caps["get_system_info"] == "ok"


def test_network_never_exposes_mac_or_virtual_interfaces(thinkpad):
    r = call_tool(thinkpad, "get_network_adapters")
    text = json.dumps(r)
    assert "3c:52:82" not in text
    assert {a["interface"] for a in r["data"]["adapters"]} == {"wlp2s0", "enp3s0"}  # no lo, no docker0
    assert all("/address" not in s for s in r["sources"])


def test_serial_numbers_are_never_read(thinkpad):
    r = call_tool(thinkpad, "get_system_info")
    assert "PF3XYZ12" not in json.dumps(r)
    with pytest.raises(PermissionError):
        thinkpad.read("/sys/class/dmi/id/product_serial")


@pytest.mark.parametrize("path", ["/etc/shadow", "/home/user/.ssh/id_rsa", "/proc/self/environ",
                                  "/sys/class/power_supply/../../../etc/passwd", "relative/path",
                                  "/sys/class/net/wlp2s0/address"])
def test_read_allow_list_blocks(path):
    assert not S.path_allowed(path)


def test_unknown_tool_and_bad_arguments_are_rejected(thinkpad):
    with pytest.raises(ToolError):
        call_tool(thinkpad, "run_shell", {"cmd": "rm -rf /"})
    with pytest.raises(ToolError):
        call_tool(thinkpad, "get_cpu_info", {"extra": 1})
    with pytest.raises(ToolError):
        call_tool(thinkpad, "get_disk_usage", {"mountpoint": "/; rm -rf /"})
    with pytest.raises(ToolError):
        call_tool(thinkpad, "get_disk_usage", {"mountpoint": "$(reboot)"})
    with pytest.raises(ToolError):
        validate_args(TOOLS["get_storage_info"].parameters, {"include_partitions": "yes"})
    assert call_tool(thinkpad, "get_disk_usage", '{"mountpoint": "/"}')["data"]["filesystems"][0]["mountpoint"] == "/"


def test_commands_are_fixed_argv_never_shell(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, 0, stdout=b'{"lscpu": []}', stderr=b"")

    monkeypatch.setattr(S.subprocess, "run", fake_run)
    live = S.LiveSystem.__new__(S.LiveSystem)
    live.is_root, live.sudo_allowed = False, False
    live._paths = {"lscpu": "/usr/bin/lscpu", "dmidecode": "/usr/sbin/dmidecode", "sudo": "/usr/bin/sudo"}
    assert live.run("lscpu").ok
    argv, kw = calls[0]
    assert argv == ["/usr/bin/lscpu", "-J"]
    assert kw["shell"] is False and kw["env"]["LC_ALL"] == "C" and kw["stdin"] == subprocess.DEVNULL
    with pytest.raises(KeyError):
        live.run("bash")  # not in the allow-list

    # root-only command without --sudo: not even attempted
    res = live.run("dmidecode_memory")
    assert res.status == "permission_denied" and len(calls) == 1
    # with --sudo: non-interactive sudo of the exact binary path
    live.sudo_allowed = True
    live.run("dmidecode_memory")
    assert calls[-1][0] == ["/usr/bin/sudo", "-n", "/usr/sbin/dmidecode", "-t", "16", "-t", "17"]


def test_sudo_password_prompt_is_reported_as_permission(monkeypatch):
    monkeypatch.setattr(S.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(
        argv, 1, stdout=b"", stderr=b"sudo: a password is required"))
    live = S.LiveSystem.__new__(S.LiveSystem)
    live.is_root, live.sudo_allowed = False, True
    live._paths = {"dmidecode": "/usr/sbin/dmidecode", "sudo": "/usr/bin/sudo"}
    assert live.run("dmidecode_memory").status == "permission_denied"


def test_binaries_resolve_only_from_trusted_dirs(monkeypatch, tmp_path):
    evil = tmp_path / "lscpu"
    evil.write_text("#!/bin/sh\necho pwned")
    evil.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    resolved = S.LiveSystem._resolve("lscpu")
    assert resolved is None or not resolved.startswith(str(tmp_path))


def test_redact_scrubs_identifiers():
    out = redact({"note": "mac 3c:52:82:aa:bb:cc ip 192.168.1.20", "serial_number": "X", "ok": [1, "a"]})
    assert out == {"note": "mac [mac] ip [ip]", "ok": [1, "a"]}
