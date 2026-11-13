from hwqa import parsers as P

DMI_SOLDERED = """# dmidecode 3.4
Handle 0x0010, DMI type 16, 23 bytes
Physical Memory Array
\tLocation: System Board Or Motherboard
\tUse: System Memory
\tMaximum Capacity: 40 GB
\tNumber Of Devices: 2

Handle 0x0011, DMI type 17, 92 bytes
Memory Device
\tSize: 8 GB
\tForm Factor: Row Of Chips
\tLocator: Onboard Memory
\tType: DDR4
\tSpeed: 3200 MT/s
\tManufacturer: Micron
\tSerial Number: 00000000

Handle 0x0012, DMI type 17, 92 bytes
Memory Device
\tSize: 8192 MB
\tForm Factor: SODIMM
\tLocator: DIMM 0
\tType: DDR4
\tSpeed: 3200 MT/s
\tManufacturer: Kingston
\tSerial Number: 12AB34CD
"""


def test_parse_size_units():
    assert P.parse_size("16 GB") == 16 * 1024**3
    assert P.parse_size("8192 MB") == 8 * 1024**3
    assert P.parse_size("No Module Installed") is None
    assert P.parse_size("") is None


def test_dmidecode_soldered_and_serials_dropped():
    out = P.parse_dmidecode_memory(DMI_SOLDERED)
    assert out["arrays"][0]["max_capacity_bytes"] == 40 * 1024**3
    soldered, socket = out["devices"]
    assert soldered["soldered"] is True and socket["soldered"] is False
    assert socket["size_bytes"] == 8 * 1024**3
    assert all("serial" not in k.lower() for d in out["devices"] for k in d)


def test_dmidecode_empty_slot():
    text = "Handle 0x1, DMI type 17, 92 bytes\nMemory Device\n\tSize: No Module Installed\n\tLocator: DIMM 1\n\tType: Unknown\n"
    dev = P.parse_dmidecode_memory(text)["devices"][0]
    assert dev["installed"] is False and dev["size_bytes"] is None and dev["type"] is None


def test_lscpu_flat_and_nested_formats(thinkpad, legion):
    flat = P.parse_lscpu_json(thinkpad.run("lscpu").stdout)
    nested = P.parse_lscpu_json(legion.run("lscpu").stdout)
    for f in (flat, nested):
        assert f["Core(s) per socket"] == "8" and f["CPU(s)"] == "16" and "avx2" in f["Flags"]
    assert nested["L3"] == "16 MiB (1 instance)"


def test_lsblk_string_and_bool_variants(thinkpad, legion):
    a = P.parse_lsblk_json(thinkpad.run("lsblk").stdout)  # newer lsblk: real booleans and ints
    b = P.parse_lsblk_json(legion.run("lsblk").stdout)    # older lsblk: "0"/"1" and string sizes
    assert a[0]["rotational"] is False and isinstance(a[0]["size_bytes"], int)
    sda = next(d for d in b if d["name"] == "sda")
    assert sda["rotational"] is True and sda["size_bytes"] == 1000204886016


def test_classify_disk():
    assert P.classify_disk({"name": "nvme0n1", "transport": "nvme"}) == "NVMe SSD"
    assert P.classify_disk({"name": "sda", "transport": "sata", "rotational": True}) == "HDD"
    assert P.classify_disk({"name": "sda", "transport": "sata", "rotational": False}) == "SATA SSD"
    assert P.classify_disk({"name": "sdb", "transport": "usb", "rotational": False}) == "USB drive"
    assert P.classify_disk({"name": "mmcblk0"}) == "eMMC/SD"
    assert P.classify_disk({"name": "vda", "rotational": True}) == "virtual disk"


def test_lspci_ids_split(legion):
    recs = P.parse_lspci_vmm(legion.run("lspci").stdout)
    nv = recs[0]
    assert nv["vendor"] == "NVIDIA Corporation" and nv["vendor_id"] == "10de" and nv["driver"] == "nvidia"


def test_misc_parsers():
    assert P.parse_nvidia_smi("NVIDIA GeForce RTX 3060 Laptop GPU, 6144, 545.29.06")[0]["vram_bytes"] == 6 * 1024**3
    usb = P.parse_lsusb("Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub\n"
                        "Bus 003 Device 002: ID 8087:0029 Intel Corp. AX200 Bluetooth")
    assert [u["name"] for u in usb] == ["Intel Corp. AX200 Bluetooth"]
    assert P.parse_os_release('PRETTY_NAME="Ubuntu 22.04.4 LTS"\nID=ubuntu')["ID"] == "ubuntu"
    df = P.parse_df("Filesystem Type 1B-blocks Used Avail Mounted on\n/dev/sda1 ext4 100 40 60 /mnt/my disk\n")
    assert df[0]["mountpoint"] == "/mnt/my disk"
    assert P.parse_meminfo("MemTotal:  1024 kB\nHugePages_Total: 0")["MemTotal"] == 1024 * 1024
    assert P.parse_cpuinfo("processor : 0\nmodel name : X\nphysical id : 0\ncore id : 0\n\n"
                           "processor : 1\nmodel name : X\nphysical id : 0\ncore id : 0\n") == {
        "model": "X", "vendor": None, "threads": 2, "cores": 1, "flags": []}
