"""The only code that touches the operating system.

Two layers of allow-listing keep the LLM away from the shell:

1. The model can only call *tools* (tools.py) with schema-validated arguments.
2. Tools can only run *commands* from COMMANDS below (fixed binary + fixed argv) and
   read files under READABLE_PREFIXES. Nothing is ever passed through a shell, and no
   model-provided string ever becomes part of an argv or a path.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TRUSTED_DIRS = ("/usr/sbin", "/usr/bin", "/sbin", "/bin")
SAFE_ENV = {"LC_ALL": "C", "LANG": "C", "PATH": "/usr/sbin:/usr/bin:/sbin:/bin"}
MAX_OUTPUT = 512 * 1024
MAX_FILE = 64 * 1024


@dataclass(frozen=True)
class CommandSpec:
    binary: str
    args: Tuple[str, ...]
    needs_root: bool = False
    timeout: float = 5.0
    package: str = ""  # where to get it, for "not installed" messages


COMMANDS: Dict[str, CommandSpec] = {
    "lscpu": CommandSpec("lscpu", ("-J",), package="util-linux"),
    "lsblk": CommandSpec("lsblk", ("-J", "-b", "-o", "NAME,TYPE,SIZE,MODEL,ROTA,TRAN,RM,MOUNTPOINT,FSTYPE"), package="util-linux"),
    "lspci": CommandSpec("lspci", ("-vmm", "-nn", "-k"), package="pciutils"),
    "lsusb": CommandSpec("lsusb", (), package="usbutils"),
    "df": CommandSpec("df", ("-B1", "--output=source,fstype,size,used,avail,target",
                             "-x", "tmpfs", "-x", "devtmpfs", "-x", "squashfs", "-x", "overlay", "-x", "efivarfs"),
                      package="coreutils"),
    "dmidecode_memory": CommandSpec("dmidecode", ("-t", "16", "-t", "17"), needs_root=True, package="dmidecode"),
    "nvidia_smi": CommandSpec("nvidia-smi", ("--query-gpu=name,memory.total,driver_version",
                                             "--format=csv,noheader,nounits"), timeout=10.0, package="NVIDIA driver"),
}

READABLE_PREFIXES = (
    "/proc/cpuinfo", "/proc/meminfo", "/proc/sys/kernel/osrelease", "/etc/os-release",
    "/sys/class/dmi/id/", "/sys/class/power_supply/", "/sys/class/net/", "/sys/class/thermal/",
    "/sys/block/", "/sys/devices/system/cpu/",
)
# Identifiers we never read, even where the kernel exposes them without root.
DENIED_NAMES = {"product_serial", "board_serial", "chassis_serial", "product_uuid", "address", "serial_number"}


@dataclass
class CommandResult:
    command: str
    ok: bool
    stdout: str = ""
    error: str = ""
    status: str = "ok"  # ok | missing | permission_denied | failed | timeout

    @property
    def display(self) -> str:
        spec = COMMANDS[self.command]
        return " ".join((spec.binary, *spec.args))


def path_allowed(path: str) -> bool:
    if ".." in Path(path).parts or not path.startswith("/"):
        return False
    if Path(path).name in DENIED_NAMES:
        return False
    return any(path == p or path.startswith(p) for p in READABLE_PREFIXES)


class SystemSource:
    """Interface the tools use. LiveSystem talks to the OS; FixtureSystem replays recordings."""

    is_root: bool = False
    sudo_allowed: bool = False

    def run(self, command: str) -> CommandResult:
        raise NotImplementedError

    def read(self, path: str) -> Optional[str]:
        raise NotImplementedError

    def listdir(self, path: str) -> List[str]:
        raise NotImplementedError

    def has(self, binary: str) -> bool:
        raise NotImplementedError

    # shared helpers ----------------------------------------------------------
    def read_value(self, path: str) -> Optional[str]:
        text = self.read(path)
        return text.strip() if text is not None else None

    def can_run(self, command: str) -> Tuple[bool, str]:
        spec = COMMANDS[command]
        if not self.has(spec.binary):
            return False, "missing"
        if spec.needs_root and not (self.is_root or self.sudo_allowed):
            return False, "needs_root"
        return True, "ok"


class LiveSystem(SystemSource):
    def __init__(self, sudo_allowed: bool = False):
        self.is_root = hasattr(os, "geteuid") and os.geteuid() == 0
        self.sudo_allowed = sudo_allowed
        self._paths = {b: self._resolve(b) for b in {s.binary for s in COMMANDS.values()} | {"sudo"}}

    @staticmethod
    def _resolve(binary: str) -> Optional[str]:
        # Only trusted system directories: a writable dir earlier in $PATH must not be able
        # to substitute its own "lscpu".
        found = shutil.which(binary, path=os.pathsep.join(TRUSTED_DIRS))
        return os.path.realpath(found) if found else None

    def has(self, binary: str) -> bool:
        return self._paths.get(binary) is not None

    def run(self, command: str) -> CommandResult:
        spec = COMMANDS[command]  # KeyError for anything not allow-listed
        exe = self._paths.get(spec.binary)
        if exe is None:
            return CommandResult(command, False, error=f"{spec.binary} is not installed ({spec.package})", status="missing")
        argv = [exe, *spec.args]
        if spec.needs_root and not self.is_root:
            if not (self.sudo_allowed and self._paths.get("sudo")):
                return CommandResult(command, False, error=f"{spec.binary} needs root", status="permission_denied")
            # -n: never prompt. Works only if the admin installed the scoped sudoers rule
            # (contrib/sudoers.d/hwqa); otherwise it fails fast instead of hanging.
            argv = [self._paths["sudo"], "-n", *argv]
        try:
            proc = subprocess.run(argv, shell=False, env=SAFE_ENV, stdin=subprocess.DEVNULL,
                                  capture_output=True, timeout=spec.timeout, check=False)
        except subprocess.TimeoutExpired:
            return CommandResult(command, False, error=f"timed out after {spec.timeout}s", status="timeout")
        except OSError as exc:
            return CommandResult(command, False, error=str(exc), status="failed")
        out = proc.stdout[:MAX_OUTPUT].decode("utf-8", "replace")
        err = proc.stderr[:4096].decode("utf-8", "replace").strip()
        if proc.returncode != 0:
            lowered = err.lower()
            denied = any(s in lowered for s in ("permission denied", "must be root", "password is required",
                                                 "not in the sudoers", "a terminal is required"))
            return CommandResult(command, False, stdout=out, error=err or f"exit code {proc.returncode}",
                                 status="permission_denied" if denied else "failed")
        return CommandResult(command, True, stdout=out)

    def read(self, path: str) -> Optional[str]:
        if not path_allowed(path):
            raise PermissionError(f"path not allow-listed: {path}")
        real = os.path.realpath(path)
        # /sys is full of symlinks; the target must stay inside /sys or /proc.
        if not (real.startswith("/sys/") or real.startswith("/proc/") or real == "/etc/os-release"
                or real.startswith("/usr/lib/os-release")):
            return None
        try:
            with open(real, "rb") as fh:
                return fh.read(MAX_FILE).decode("utf-8", "replace")
        except OSError:
            return None

    def listdir(self, path: str) -> List[str]:
        if not path_allowed(path.rstrip("/") + "/"):
            raise PermissionError(f"path not allow-listed: {path}")
        try:
            return sorted(os.listdir(path))
        except OSError:
            return []


class FixtureSystem(SystemSource):
    """Replays a recorded machine (see fixtures/README.md). Used for tests, demos on
    non-Linux hosts, and reproducing bug reports without access to the machine."""

    def __init__(self, root: str, sudo_allowed: bool = False):
        self.root = Path(root)
        meta_path = self.root / "meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.is_root = bool(self.meta.get("root", False))
        self.sudo_allowed = sudo_allowed and bool(self.meta.get("sudo_rule", True))
        self.missing = set(self.meta.get("missing_binaries", []))

    def has(self, binary: str) -> bool:
        return binary not in self.missing

    def run(self, command: str) -> CommandResult:
        spec = COMMANDS[command]
        if not self.has(spec.binary):
            return CommandResult(command, False, error=f"{spec.binary} is not installed ({spec.package})", status="missing")
        if spec.needs_root and not (self.is_root or self.sudo_allowed):
            return CommandResult(command, False, error=f"{spec.binary} needs root", status="permission_denied")
        out = self.root / "commands" / f"{command}.txt"
        if not out.exists():
            return CommandResult(command, False, error="command failed (no output recorded)", status="failed")
        return CommandResult(command, True, stdout=out.read_text(encoding="utf-8"))

    def _local(self, path: str) -> Path:
        return self.root / "files" / path.lstrip("/")

    def read(self, path: str) -> Optional[str]:
        if not path_allowed(path):
            raise PermissionError(f"path not allow-listed: {path}")
        p = self._local(path)
        return p.read_text(encoding="utf-8") if p.is_file() else None

    def listdir(self, path: str) -> List[str]:
        if not path_allowed(path.rstrip("/") + "/"):
            raise PermissionError(f"path not allow-listed: {path}")
        p = self._local(path)
        return sorted(x.name for x in p.iterdir()) if p.is_dir() else []
