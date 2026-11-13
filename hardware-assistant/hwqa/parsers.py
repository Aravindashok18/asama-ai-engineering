"""Turn raw command output and /proc, /sys text into normalised, typed data.

Rules: sizes become integer bytes, speeds integer MHz / MT/s, booleans real booleans,
"Unknown" / "Not Specified" become None. Every parser tolerates the format differences
between util-linux / dmidecode versions that ship with common distros.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

UNIT = {"b": 1, "kb": 1024, "kib": 1024, "mb": 1024**2, "mib": 1024**2, "gb": 1024**3, "gib": 1024**3,
        "tb": 1024**4, "tib": 1024**4}
NULLISH = {"", "unknown", "not specified", "not provided", "none", "to be filled by o.e.m.", "default string",
           "not available", "n/a", "[not supported]", "0000"}


def clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    return None if v.lower() in NULLISH else v


def parse_size(text: Optional[str]) -> Optional[int]:
    """'16 GB' / '16384 MB' / '64 GB' -> bytes (binary units, as dmidecode and /proc use)."""
    if not text:
        return None
    m = re.match(r"^\s*([\d.]+)\s*([kmgt]i?b|b)?\s*$", text.strip(), re.I)
    if not m:
        return None
    return int(float(m.group(1)) * UNIT[(m.group(2) or "b").lower()])


def to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in ("1", 1, "true"):
        return True
    if value in ("0", 0, "false"):
        return False
    return None


def to_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------- CPU


def parse_lscpu_json(text: str) -> Dict[str, str]:
    """lscpu -J; newer util-linux nests some fields under `children`."""
    out: Dict[str, str] = {}

    def walk(items):
        for item in items:
            key = item.get("field", "").rstrip(":").strip()
            if key and item.get("data") is not None:
                out.setdefault(key, str(item["data"]).strip())
            walk(item.get("children", []))

    walk(json.loads(text).get("lscpu", []))
    return out


def parse_cpuinfo(text: str) -> Dict[str, Any]:
    """Fallback when lscpu is missing: /proc/cpuinfo."""
    blocks = [b for b in text.strip().split("\n\n") if b.strip()]
    procs = []
    for block in blocks:
        d = {}
        for line in block.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
        procs.append(d)
    if not procs:
        return {}
    first = procs[0]
    physical = {(p.get("physical id"), p.get("core id")) for p in procs}
    return {
        "model": first.get("model name") or first.get("Model") or first.get("Hardware"),
        "vendor": first.get("vendor_id") or first.get("CPU implementer"),
        "threads": len(procs),
        "cores": len(physical) if all(k != (None, None) for k in physical) else None,
        "flags": (first.get("flags") or first.get("Features") or "").split(),
    }


# -------------------------------------------------------------------- memory


def parse_meminfo(text: str) -> Dict[str, int]:
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+(?:\(\w+\))?):\s+(\d+)\s*(kB)?", line)
        if m:
            out[m.group(1)] = int(m.group(2)) * (1024 if m.group(3) else 1)
    return out


def _dmi_records(text: str) -> List[Dict[str, Any]]:
    records = []
    for chunk in re.split(r"\n(?=Handle 0x)", text):
        lines = chunk.strip().splitlines()
        if not lines or not lines[0].startswith("Handle"):
            continue
        m = re.search(r"DMI type (\d+)", lines[0])
        rec: Dict[str, Any] = {"_type": int(m.group(1)) if m else None, "_name": lines[1].strip() if len(lines) > 1 else ""}
        for line in lines[2:]:
            if ":" in line and line.startswith("\t") and not line.startswith("\t\t"):
                k, v = line.strip().split(":", 1)
                rec[k.strip()] = v.strip()
        records.append(rec)
    return records


def parse_dmidecode_memory(text: str) -> Dict[str, Any]:
    """dmidecode -t 16 -t 17 -> memory arrays and per-slot devices (serials dropped)."""
    arrays, devices = [], []
    for rec in _dmi_records(text):
        if rec["_type"] == 16:
            if "system memory" not in rec.get("Use", "System Memory").lower():
                continue  # skip cache / flash arrays
            arrays.append({
                "max_capacity_bytes": parse_size(rec.get("Maximum Capacity")),
                "slots": to_int(rec.get("Number Of Devices")),
                "error_correction": clean(rec.get("Error Correction Type")),
            })
        elif rec["_type"] == 17:
            size_raw = rec.get("Size", "")
            installed = bool(size_raw) and "no module" not in size_raw.lower() and parse_size(size_raw) not in (None, 0)
            form = clean(rec.get("Form Factor"))
            speed = re.match(r"(\d+)", rec.get("Configured Memory Speed") or rec.get("Configured Clock Speed") or rec.get("Speed") or "")
            devices.append({
                "locator": clean(rec.get("Locator")),
                "bank": clean(rec.get("Bank Locator")),
                "installed": installed,
                "size_bytes": parse_size(size_raw) if installed else None,
                "type": clean(rec.get("Type")) if installed else None,
                "form_factor": form,
                "speed_mts": int(speed.group(1)) if installed and speed else None,
                "manufacturer": clean(rec.get("Manufacturer")) if installed else None,
                "part_number": clean(rec.get("Part Number")) if installed else None,
                # "Row Of Chips" = memory soldered to the board; it occupies no socket.
                "soldered": (form or "").lower() in ("row of chips", "chip") or "onboard" in (rec.get("Locator") or "").lower(),
            })
    return {"arrays": arrays, "devices": devices}


# ------------------------------------------------------------------- storage


def parse_lsblk_json(text: str) -> List[Dict[str, Any]]:
    def norm(d: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": d.get("name"),
            "type": d.get("type"),
            "size_bytes": to_int(d.get("size")),
            "model": clean(d.get("model")),
            "rotational": to_bool(d.get("rota")),
            "transport": clean(d.get("tran")),
            "removable": to_bool(d.get("rm")),
            "mountpoint": d.get("mountpoint"),
            "fstype": d.get("fstype"),
            "children": [norm(c) for c in d.get("children", [])],
        }

    return [norm(d) for d in json.loads(text).get("blockdevices", [])]


def classify_disk(dev: Dict[str, Any]) -> str:
    name, tran = dev.get("name") or "", (dev.get("transport") or "").lower()
    if tran == "nvme" or name.startswith("nvme"):
        return "NVMe SSD"
    if name.startswith("mmcblk"):
        return "eMMC/SD"
    if tran == "usb":
        return "USB drive"
    if name.startswith(("vd", "xvd")):
        return "virtual disk"
    if dev.get("rotational") is True:
        return "HDD"
    if dev.get("rotational") is False:
        return "SATA SSD" if tran in ("sata", "ata", "") else "SSD"
    return "unknown"


def parse_df(text: str) -> List[Dict[str, Any]]:
    rows = []
    for line in text.strip().splitlines()[1:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        source, fstype, size, used, avail = parts[:5]
        target = " ".join(parts[5:])
        rows.append({"device": source, "fstype": fstype, "size_bytes": int(size), "used_bytes": int(used),
                     "available_bytes": int(avail), "mountpoint": target})
    return rows


# ---------------------------------------------------------------------- PCI


def parse_lspci_vmm(text: str) -> List[Dict[str, Any]]:
    """lspci -vmm -nn -k: blank-line separated records, 'Key:\\tValue [id]'."""
    out = []
    for block in text.strip().split("\n\n"):
        rec: Dict[str, Any] = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip()
            m = re.match(r"^(.*?)\s*\[([0-9a-fA-F]{4})\]$", v)
            if m and k in ("Class", "Vendor", "Device", "SVendor", "SDevice"):
                rec[k.lower()] = m.group(1)
                rec[k.lower() + "_id"] = m.group(2).lower()
            else:
                rec[k.lower()] = v
        if rec:
            out.append(rec)
    return out


def parse_nvidia_smi(text: str) -> List[Dict[str, Any]]:
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "vram_bytes": to_int(parts[1]) * 1024**2 if to_int(parts[1]) else None,
                         "driver_version": parts[2]})
    return gpus


def parse_lsusb(text: str) -> List[Dict[str, Any]]:
    out = []
    for line in text.strip().splitlines():
        m = re.match(r"Bus (\d+) Device (\d+): ID ([0-9a-f]{4}):([0-9a-f]{4})\s*(.*)$", line.strip())
        if m:
            name = m.group(5).strip()
            if "root hub" in name.lower():
                continue
            out.append({"bus": int(m.group(1)), "vendor_id": m.group(3), "product_id": m.group(4), "name": name or None})
    return out


# ------------------------------------------------------------------------ OS


def parse_os_release(text: str) -> Dict[str, str]:
    out = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"')
    return out
