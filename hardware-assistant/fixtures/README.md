# Recorded machines

Each folder replays one machine so the assistant can be tested and demoed anywhere.

```
<machine>/
  meta.json         description, whether it was recorded as root, whether a sudo rule exists,
                    and which binaries are missing
  commands/<id>.txt stdout of the allow-listed command with that id (see hwqa/system.py COMMANDS)
  files/...         /proc, /sys and /etc files at their real paths
```

| Machine | What it exercises |
|---|---|
| `thinkpad-e14` | the brief's example: 16 GB in 1 of 2 slots, 64 GB max; NVMe; AMD iGPU; battery; non-root with a sudo rule |
| `legion-5` | both RAM slots full, NVIDIA + AMD GPUs with `nvidia-smi`, HDD + NVMe, nested `lscpu -J`, older `lsblk` string fields; recorded as root |
| `minimal-vm` | Debian cloud VM without `dmidecode`, `lspci`, `lsusb`, `lsblk`; no battery, no thermal zones, no clock speed |

The files include a MAC address and a serial number on purpose: tests check that hwqa never reads them.

To record a new machine, copy the outputs of the commands in `COMMANDS` (with `LC_ALL=C`) into
`commands/`, and the files the tools read into `files/`. Remove serial numbers and addresses first.
