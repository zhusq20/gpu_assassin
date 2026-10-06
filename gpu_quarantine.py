#!/usr/bin/python3
"""Quarantine this incident's failed GPU by UUID; never match bus address alone."""
from pathlib import Path
import subprocess
import re
import os

UUID='GPU-99021681-ed48-34fb-6701-934f2747f190'
assert os.geteuid()==0
for info in Path('/proc/driver/nvidia/gpus').glob('*/information'):
    data=info.read_text()
    if UUID not in data:
        continue
    bus=info.parent.name
    minor=int(re.search(r'Device Minor:\s+(\d+)',data)[1])
    dev=Path('/sys/bus/pci/devices')/bus
    assert (dev/'vendor').read_text().strip()=='0x10de'
    assert (dev/'device').read_text().strip()=='0x2bb5'
    p=subprocess.run(['/usr/bin/fuser','-v',f'/dev/nvidia{minor}'],capture_output=True,text=True)
    if p.returncode!=1:
        raise SystemExit('Failed GPU still has device users; refusing to unbind')
    (dev/'driver_override').write_text('none')
    (dev/'driver'/'unbind').write_text(bus)
    print('QUARANTINED',UUID,bus,flush=True)
    break
else:
    print('Failed GPU not bound to NVIDIA, already excluded or absent',flush=True)
