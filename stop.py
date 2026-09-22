#!/usr/bin/env python3
"""Stop the claimer: unload the daemon, then shut down ONLY the CODM AVD.

Exists because doing this by hand is dangerous. `adb emu kill` refuses to
act when several emulators are attached, and the obvious workaround --
looping over `adb devices` and killing each serial -- takes down every
emulator on the machine. That is exactly how Pixel_8 got killed on
2026-09-23 alongside CODM_Pixel9.

Other AVDs are never touched here: each serial is resolved to its AVD name
first, and anything that is not LOCKED_AVD is skipped and reported.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.adb import AdbDevice, LOCKED_AVD  # noqa: E402


def main() -> int:
    subprocess.run(["launchctl", "bootout", f"gui/{subprocess.run(['id','-u'],capture_output=True,text=True).stdout.strip()}/com.codm-dmz-claimer"],
                   check=False, capture_output=True)
    for _ in range(6):
        listed = subprocess.run(["launchctl", "list"], capture_output=True, text=True).stdout
        if "com.codm-dmz-claimer" not in listed:
            print("daemon: stopped")
            break
        time.sleep(3)
    else:
        print("daemon: still loaded — check `launchctl list | grep codm`")

    killed = skipped = 0
    for serial in AdbDevice.list_devices():
        if not serial.startswith("emulator-"):
            continue
        avd = AdbDevice.emulator_avd_name(serial)
        if avd == LOCKED_AVD:
            subprocess.run(["adb", "-s", serial, "emu", "kill"], check=False,
                           capture_output=True, timeout=15)
            print(f"stopped {serial} ({avd})")
            killed += 1
        else:
            print(f"left alone: {serial} ({avd or 'unknown AVD'})")
            skipped += 1
    if not killed:
        print(f"no {LOCKED_AVD} emulator was running")
    if skipped:
        print(f"{skipped} other emulator(s) untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
