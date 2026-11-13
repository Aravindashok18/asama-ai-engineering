"""The tool registry: the only capabilities the language model has.

Each tool gathers one area of hardware facts, parses and normalises them, and returns
  {"data": {...}, "unavailable": [...], "sources": [...]}
`unavailable` says exactly what could not be read, why, and how the user can enable it,
so the model can say "I don't know X because Y" instead of guessing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from . import parsers as P
from .system import COMMANDS, CommandResult, SystemSource

GiB = 1024**3


def gib(n: Optional[int], digits: int = 1) -> Optional[float]:
    return round(n / GiB, digits) if n is not None else None


def human_disk(n: Optional[int]) -> Optional[str]:
    """Disks are sold in decimal units, so report them that way (512 GB, not 476.9 GiB)."""
    if n is None:
        return None
    if n >= 10**12:
        return f"{n / 10**12:.2f} TB"
    if n >= 10**9:
        return f"{round(n / 10**9)} GB"
    return f"{round(n / 10**6)} MB"


class Collector:
    """Accumulates data + provenance + gaps for one tool call."""

    def __init__(self, system: SystemSource):
        self.sys = system
        self.data: Dict[str, Any] = {}
        self.unavailable: List[Dict[str, str]] = []
        self.sources: List[str] = []

    def run(self, command: str) -> CommandResult:
        res = self.sys.run(command)
        self.sources.append(res.display + ("" if res.ok else f"  [{res.status}]"))
        return res

    def read(self, path: str) -> Optional[str]:
        text = self.sys.read(path)
        if text is not None:
            self.sources.append(path)
        return text

    def value(self, path: str) -> Optional[str]:
        text = self.read(path)
        return P.clean(text) if text is not None else None

    def gap(self, what: str, res: Optional[CommandResult] = None, reason: str = "", fix: str = "") -> None:
        if res is not None:
            spec = COMMANDS[res.command]
            if res.status == "missing":
                reason = reason or f"`{spec.binary}` is not installed"
                fix = fix or f"install the `{spec.package}` package"
            elif res.status == "permission_denied":
                reason = reason or f"`{spec.binary}` needs root"
                fix = fix or ("restart with --sudo after installing contrib/sudoers.d/hwqa "
                              "(lets only this exact command run as root), or run hwqa as root")
            else:
                reason = reason or f"`{spec.binary}` failed: {res.error[:120]}"
        self.unavailable.append({"what": what, "reason": reason, "how_to_enable": fix})

    def result(self) -> Dict[str, Any]:
        return {"data": self.data, "unavailable": self.unavailable, "sources": self.sources}


# ======================================================================== tools


def system_info(c: Collector) -> None:
    dmi = {k: c.value(f"/sys/class/dmi/id/{k}") for k in (
        "sys_vendor", "product_name", "product_version", "product_family", "board_vendor", "board_name",
        "bios_vendor", "bios_version", "bios_date", "chassis_type")}
    chassis = {"8": "portable", "9": "laptop", "10": "notebook", "14": "sub-notebook", "31": "convertible",
               "32": "detachable", "3": "desktop", "4": "desktop", "6": "mini tower", "7": "tower",
               "13": "all-in-one", "35": "mini PC", "17": "server", "23": "rack server"}
    c.data["manufacturer"] = dmi["sys_vendor"]
    # Lenovo puts the marketing name in product_version and a type code in product_name
    c.data["model"] = (dmi["product_version"] if (dmi["sys_vendor"] or "").upper() == "LENOVO" and dmi["product_version"]
                       else dmi["product_name"])
    c.data["product_code"] = dmi["product_name"]
    c.data["family"] = dmi["product_family"]
    c.data["motherboard"] = " ".join(x for x in (dmi["board_vendor"], dmi["board_name"]) if x) or None
    c.data["bios"] = {"vendor": dmi["bios_vendor"], "version": dmi["bios_version"], "date": dmi["bios_date"]}
    c.data["form_factor"] = chassis.get(dmi["chassis_type"] or "", None)
    if not any(dmi.values()):
        c.gap("manufacturer, model and BIOS", reason="/sys/class/dmi/id is not available (VM or non-x86 board)")
    osr = c.read("/etc/os-release")
    if osr:
        o = P.parse_os_release(osr)
        c.data["os"] = o.get("PRETTY_NAME") or o.get("NAME")
        c.data["distro_id"] = o.get("ID")
    c.data["kernel"] = c.value("/proc/sys/kernel/osrelease")


def cpu_info(c: Collector) -> None:
    res = c.run("lscpu")
    if res.ok:
        f = P.parse_lscpu_json(res.stdout)
        sockets = P.to_int(f.get("Socket(s)")) or 1
        per_socket = P.to_int(f.get("Core(s) per socket"))
        flags = (f.get("Flags") or "").split()
        max_mhz = f.get("CPU max MHz")
        c.data.update({
            "model": f.get("Model name"),
            "vendor": f.get("Vendor ID"),
            "architecture": f.get("Architecture"),
            "sockets": sockets,
            "cores": per_socket * sockets if per_socket else None,
            "threads": P.to_int(f.get("CPU(s)")),
            "threads_per_core": P.to_int(f.get("Thread(s) per core")),
            "max_mhz": int(float(max_mhz)) if max_mhz else None,
            "min_mhz": int(float(f["CPU min MHz"])) if f.get("CPU min MHz") else None,
            # flat lscpu says "L2 cache", nested (util-linux >= 2.38) says "L2"
            "cache": {k.replace(" cache", ""): v for k, v in f.items() if re.match(r"^L\d[di]?( cache)?$", k)},
            "virtualization": f.get("Virtualization"),
        })
    else:
        text = c.read("/proc/cpuinfo")
        if not text:
            c.gap("CPU details", res)
            return
        info = P.parse_cpuinfo(text)
        flags = info.pop("flags", [])
        c.data.update(info)
        khz = c.value("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
        c.data["max_mhz"] = int(khz) // 1000 if khz and khz.isdigit() else None
        c.gap("cache sizes and exact core topology", res, fix="install util-linux for full details")
    c.data["instruction_sets"] = {k: k in flags for k in ("avx", "avx2", "avx512f", "sse4_2", "aes", "sha_ni")}
    if "max_mhz" in c.data and c.data["max_mhz"] is None:
        c.gap("maximum clock speed", reason="the CPU does not report it (common in VMs)")


def memory_info(c: Collector) -> None:
    meminfo = c.read("/proc/meminfo")
    if meminfo:
        m = P.parse_meminfo(meminfo)
        total, avail = m.get("MemTotal"), m.get("MemAvailable")
        c.data["usable_gb"] = gib(total)
        c.data["available_now_gb"] = gib(avail)
        c.data["in_use_gb"] = gib(total - avail) if total and avail else None
        c.data["swap_gb"] = gib(m.get("SwapTotal"))
        c.data["note_usable_vs_installed"] = ("usable_gb is what the OS can use; it is a little lower than the installed "
                                              "size because firmware and integrated graphics reserve some RAM")
    res = c.run("dmidecode_memory")
    if not res.ok:
        c.gap("installed module sizes, slot count, memory type/speed and maximum supported RAM", res)
        return
    dmi = P.parse_dmidecode_memory(res.stdout)
    devices = dmi["devices"]
    sockets = [d for d in devices if not d["soldered"]]
    soldered = [d for d in devices if d["soldered"] and d["installed"]]
    used = [d for d in sockets if d["installed"]]
    installed_bytes = sum(d["size_bytes"] or 0 for d in devices if d["installed"])
    max_cap = sum(a["max_capacity_bytes"] or 0 for a in dmi["arrays"]) or None
    first = next((d for d in devices if d["installed"]), {})

    c.data["installed_gb"] = gib(installed_bytes, 0)
    c.data["type"] = first.get("type")
    c.data["speed_mts"] = first.get("speed_mts")
    c.data["modules"] = [{
        "slot": d["locator"], "installed": d["installed"], "size_gb": gib(d["size_bytes"], 0),
        "type": d["type"], "form_factor": d["form_factor"], "speed_mts": d["speed_mts"],
        "manufacturer": d["manufacturer"], "part_number": d["part_number"], "soldered": d["soldered"],
    } for d in devices]

    upgrade: Dict[str, Any] = {
        "total_slots": len(sockets), "used_slots": len(used), "free_slots": len(sockets) - len(used),
        "soldered_gb": gib(sum(d["size_bytes"] for d in soldered), 0) if soldered else 0,
        "max_supported_gb": gib(max_cap, 0),
    }
    if max_cap and sockets:
        socket_budget = max_cap - sum(d["size_bytes"] for d in soldered)
        per_slot = socket_budget / len(sockets)
        upgrade["max_per_slot_gb"] = gib(int(per_slot), 0)
        addable = int(min(upgrade["free_slots"] * per_slot, max_cap - installed_bytes))
        upgrade["max_addable_gb_keeping_current_modules"] = gib(addable, 0)
        upgrade["max_total_gb_keeping_current_modules"] = gib(installed_bytes + addable, 0)
        upgrade["max_total_gb_if_replacing_modules"] = gib(max_cap, 0)
        upgrade["notes"] = [
            "maximum is what the firmware (SMBIOS) reports; vendors sometimes certify more or less, so check the "
            "laptop's service manual before buying",
            "per-slot maximum is estimated as maximum / number of sockets",
        ]
        if first.get("type"):
            upgrade["buy"] = f"{first['type']} {first.get('form_factor') or ''}".strip() + (
                f", {first['speed_mts']} MT/s or faster" if first.get("speed_mts") else "")
    elif not max_cap:
        c.gap("maximum supported RAM", reason="firmware does not report a maximum capacity")
    if not sockets:
        upgrade["notes"] = ["all memory is soldered to the motherboard; it cannot be upgraded"]
    c.data["upgrade"] = upgrade


def storage_info(c: Collector, include_partitions: bool = True) -> None:
    res = c.run("lsblk")
    disks: List[Dict[str, Any]] = []
    if res.ok:
        for d in P.parse_lsblk_json(res.stdout):
            if d["type"] != "disk" or (d["name"] or "").startswith(("zram", "loop", "ram")):
                continue
            disk = {"device": f"/dev/{d['name']}", "model": d["model"], "size": human_disk(d["size_bytes"]),
                    "size_bytes": d["size_bytes"], "kind": P.classify_disk(d), "transport": d["transport"],
                    "removable": d["removable"]}
            if include_partitions:
                disk["partitions"] = [{"device": f"/dev/{p['name']}", "size": human_disk(p["size_bytes"]),
                                       "fstype": p["fstype"], "mountpoint": p["mountpoint"]} for p in d["children"]]
            disks.append(disk)
    else:
        for name in c.sys.listdir("/sys/block"):
            if name.startswith(("loop", "ram", "zram", "dm-", "sr")):
                continue
            sectors = c.value(f"/sys/block/{name}/size")
            rota = c.value(f"/sys/block/{name}/queue/rotational")
            dev = {"name": name, "rotational": P.to_bool(rota), "transport": None}
            size = int(sectors) * 512 if sectors and sectors.isdigit() else None
            disks.append({"device": f"/dev/{name}", "model": c.value(f"/sys/block/{name}/device/model"),
                          "size": human_disk(size), "size_bytes": size, "kind": P.classify_disk(dev),
                          "transport": None, "removable": P.to_bool(c.value(f"/sys/block/{name}/removable"))})
        c.gap("partition layout and transport type", res, fix="install util-linux for lsblk")
    c.data["disks"] = disks
    if not disks:
        c.gap("disks", reason="no block devices found")


def disk_usage(c: Collector, mountpoint: Optional[str] = None) -> None:
    res = c.run("df")
    if not res.ok:
        c.gap("free space per filesystem", res)
        return
    rows = P.parse_df(res.stdout)
    if mountpoint:
        rows = [r for r in rows if r["mountpoint"] == mountpoint]
        if not rows:
            c.gap(f"usage for {mountpoint}", reason="no mounted filesystem at that path")
    c.data["filesystems"] = [{
        "mountpoint": r["mountpoint"], "device": r["device"], "fstype": r["fstype"],
        "size": human_disk(r["size_bytes"]), "used": human_disk(r["used_bytes"]), "free": human_disk(r["available_bytes"]),
        "used_percent": round(100 * r["used_bytes"] / r["size_bytes"]) if r["size_bytes"] else None,
    } for r in rows]


GPU_CLASSES = ("vga compatible controller", "3d controller", "display controller")
PCI_VENDORS = {"8086": "Intel", "1002": "AMD", "10de": "NVIDIA", "1af4": "Red Hat (virtio)", "15ad": "VMware", "1234": "QEMU"}


def gpu_info(c: Collector) -> None:
    gpus: List[Dict[str, Any]] = []
    res = c.run("lspci")
    if res.ok:
        for r in P.parse_lspci_vmm(res.stdout):
            if (r.get("class") or "").lower() in GPU_CLASSES:
                gpus.append({"vendor": r.get("vendor"), "model": r.get("device"), "pci_slot": r.get("slot"),
                             "driver": r.get("driver"), "vendor_id": r.get("vendor_id")})
    else:
        c.gap("GPU model names", res)
    if any(g["vendor_id"] == "10de" for g in gpus):
        smi = c.run("nvidia_smi")
        if smi.ok:
            nv = P.parse_nvidia_smi(smi.stdout)
            nv_gpus = [g for g in gpus if g["vendor_id"] == "10de"]
            for g, n in zip(nv_gpus, nv):
                g.update({"name": n["name"], "vram_gb": gib(n["vram_bytes"]), "nvidia_driver": n["driver_version"]})
        else:
            c.gap("NVIDIA VRAM size and driver version", smi,
                  fix="install the proprietary NVIDIA driver (provides nvidia-smi)" if smi.status == "missing" else "")
    for g in gpus:
        g.pop("vendor_id", None)
        g["vendor"] = g["vendor"] or None
    c.data["gpus"] = gpus
    c.data["count"] = len(gpus)
    if res.ok and not gpus:
        c.gap("GPU", reason="no display controller found on the PCI bus")


def battery_info(c: Collector) -> None:
    batteries = []
    for name in c.sys.listdir("/sys/class/power_supply"):
        base = f"/sys/class/power_supply/{name}"
        if (c.value(f"{base}/type") or "").lower() != "battery":
            continue
        full = c.value(f"{base}/energy_full") or c.value(f"{base}/charge_full")
        design = c.value(f"{base}/energy_full_design") or c.value(f"{base}/charge_full_design")
        unit = "Wh" if c.value(f"{base}/energy_full") else "Ah"
        cycles = P.to_int(c.value(f"{base}/cycle_count"))
        b = {
            "name": name, "status": c.value(f"{base}/status"), "charge_percent": P.to_int(c.value(f"{base}/capacity")),
            "technology": c.value(f"{base}/technology"), "manufacturer": c.value(f"{base}/manufacturer"),
            "model": c.value(f"{base}/model_name"),
            "cycle_count": cycles if cycles else None,  # 0 usually means "not reported"
        }
        if full and design and full.isdigit() and design.isdigit() and int(design) > 0:
            b[f"full_capacity_{unit.lower()}"] = round(int(full) / 1e6, 1)
            b[f"design_capacity_{unit.lower()}"] = round(int(design) / 1e6, 1)
            b["health_percent"] = round(100 * int(full) / int(design))
        else:
            c.gap(f"{name} health (full vs design capacity)", reason="the battery does not report capacity")
        if b["cycle_count"] is None:
            c.gap(f"{name} cycle count", reason="the battery firmware does not report it")
        batteries.append(b)
    c.data["batteries"] = batteries
    ac = [n for n in c.sys.listdir("/sys/class/power_supply")
          if (c.value(f"/sys/class/power_supply/{n}/type") or "").lower() == "mains"]
    if ac:
        c.data["on_ac_power"] = c.value(f"/sys/class/power_supply/{ac[0]}/online") == "1"
    if not batteries:
        c.gap("battery", reason="no battery found (desktop, VM, or battery not detected)")


def network_info(c: Collector) -> None:
    adapters = []
    for name in c.sys.listdir("/sys/class/net"):
        if name == "lo":
            continue
        base = f"/sys/class/net/{name}"
        entries = c.sys.listdir(base)
        if "device" not in entries:
            continue  # virtual interface (docker0, veth, tun)
        uevent = c.read(f"{base}/device/uevent") or ""
        driver = next((l.split("=", 1)[1] for l in uevent.splitlines() if l.startswith("DRIVER=")), None)
        speed = P.to_int(c.value(f"{base}/speed"))
        adapters.append({"interface": name, "kind": "wifi" if ("wireless" in entries or "phy80211" in entries) else "ethernet",
                         "state": c.value(f"{base}/operstate"), "driver": driver,
                         "link_speed_mbps": speed if speed and speed > 0 else None})
    res = c.run("lspci")
    if res.ok:
        c.data["controllers"] = [f"{r.get('vendor')} {r.get('device')}".strip() for r in P.parse_lspci_vmm(res.stdout)
                                 if (r.get("class") or "").lower() in ("network controller", "ethernet controller")]
    c.data["adapters"] = adapters
    c.data["privacy"] = "MAC and IP addresses are intentionally not collected"


def temperatures(c: Collector) -> None:
    zones = []
    for z in c.sys.listdir("/sys/class/thermal"):
        if not z.startswith("thermal_zone"):
            continue
        t = P.to_int(c.value(f"/sys/class/thermal/{z}/temp"))
        if t is not None:
            zones.append({"sensor": c.value(f"/sys/class/thermal/{z}/type"), "celsius": round(t / 1000, 1)})
    c.data["sensors"] = zones
    if not zones:
        c.gap("temperatures", reason="no thermal zones exposed by the kernel")


def usb_devices(c: Collector) -> None:
    res = c.run("lsusb")
    if not res.ok:
        c.gap("USB devices", res)
        return
    c.data["devices"] = P.parse_lsusb(res.stdout)


# ===================================================================== registry


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., None]
    parameters: Dict[str, Any]
    ttl_s: float = 600.0  # cache lifetime; hardware rarely changes, sensors do

    def schema(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


NO_ARGS = {"type": "object", "properties": {}, "additionalProperties": False}

TOOLS: Dict[str, Tool] = {t.name: t for t in [
    Tool("get_system_info", "Laptop manufacturer and model, motherboard, BIOS version/date, form factor, OS/distro and kernel.",
         system_info, NO_ARGS),
    Tool("get_cpu_info", "CPU model, vendor, architecture, physical cores, threads, max/min clock (MHz), caches, "
         "virtualization and instruction-set support (AVX2, AVX-512...).", cpu_info, NO_ARGS),
    Tool("get_memory_info", "RAM: usable and available memory, installed modules per slot (size, type, speed), "
         "free slots, maximum supported RAM and a computed upgrade analysis. Use for any RAM upgrade question.",
         memory_info, NO_ARGS, ttl_s=30),
    Tool("get_storage_info", "Physical disks: model, capacity, kind (NVMe SSD / SATA SSD / HDD / USB), transport and partitions.",
         storage_info, {"type": "object", "additionalProperties": False, "properties": {
             "include_partitions": {"type": "boolean", "description": "include partitions and mount points"}}}),
    Tool("get_disk_usage", "Used and free space per mounted filesystem.", disk_usage,
         {"type": "object", "additionalProperties": False, "properties": {
             "mountpoint": {"type": "string", "maxLength": 200, "pattern": r"^/[A-Za-z0-9._/@ -]*$",
                            "description": "only this mount point, e.g. '/' or '/home'"}}}, ttl_s=10),
    Tool("get_gpu_info", "Graphics adapters: vendor, model, kernel driver; VRAM and driver version for NVIDIA GPUs.",
         gpu_info, NO_ARGS),
    Tool("get_battery_info", "Battery charge %, charging status, health (full vs design capacity), cycle count, AC power.",
         battery_info, NO_ARGS, ttl_s=10),
    Tool("get_network_adapters", "Wi-Fi and Ethernet adapters: controller model, driver, link state and speed. No addresses.",
         network_info, NO_ARGS, ttl_s=30),
    Tool("get_temperatures", "Current temperature sensor readings in Celsius.", temperatures, NO_ARGS, ttl_s=5),
    Tool("get_usb_devices", "Connected USB devices (webcam, Bluetooth, fingerprint reader, external drives...).",
         usb_devices, NO_ARGS, ttl_s=10),
]}


class ToolError(Exception):
    pass


def validate_args(schema: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Strict subset of JSON Schema: object of known scalar properties, nothing else."""
    if args in (None, ""):
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            raise ToolError("arguments must be a JSON object")
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    props = schema.get("properties", {})
    unknown = set(args) - set(props)
    if unknown:
        raise ToolError(f"unknown argument(s): {', '.join(sorted(unknown))}")
    for key, value in args.items():
        spec = props[key]
        kind = spec.get("type")
        if kind == "boolean" and not isinstance(value, bool):
            raise ToolError(f"{key} must be a boolean")
        if kind == "string":
            if not isinstance(value, str):
                raise ToolError(f"{key} must be a string")
            if len(value) > spec.get("maxLength", 10_000):
                raise ToolError(f"{key} is too long")
            if "pattern" in spec and not re.match(spec["pattern"], value):
                raise ToolError(f"{key} has invalid characters")
    return args


_MAC = re.compile(r"\b[0-9a-f]{2}(:[0-9a-f]{2}){5}\b", re.I)
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def redact(obj: Any) -> Any:
    """Defence in depth: tools do not collect identifiers, but scrub anything that slips in."""
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items() if "serial" not in k.lower() and "uuid" not in k.lower()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return _IPV4.sub("[ip]", _MAC.sub("[mac]", obj))
    return obj


def call_tool(system: SystemSource, name: str, args: Any = None) -> Dict[str, Any]:
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolError(f"unknown tool '{name}'. Available: {', '.join(TOOLS)}")
    kwargs = validate_args(tool.parameters, args)
    c = Collector(system)
    tool.fn(c, **kwargs)
    return redact(c.result())


def capability_report(system: SystemSource) -> Dict[str, str]:
    """What will work on this machine, computed at startup (shown in /tools and the system prompt)."""
    report = {}
    needs = {"get_cpu_info": ["lscpu"], "get_memory_info": ["dmidecode_memory"], "get_storage_info": ["lsblk"],
             "get_disk_usage": ["df"], "get_gpu_info": ["lspci"], "get_network_adapters": ["lspci"],
             "get_usb_devices": ["lsusb"]}
    for name in TOOLS:
        problems = []
        for cmd in needs.get(name, []):
            ok, why = system.can_run(cmd)
            if not ok:
                problems.append(f"{COMMANDS[cmd].binary} {'not installed' if why == 'missing' else 'needs root'}")
        report[name] = "ok" if not problems else "partial: " + ", ".join(problems)
    return report
