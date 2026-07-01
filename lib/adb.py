"""Thin wrapper around the `adb` CLI. We shell out instead of using a Python ADB
library because the surface we need is small and `adb` is already installed
alongside Android Studio / platform-tools.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2


# HARD LOCK: this project drives exactly one AVD, by name. The picker refuses
# to attach to anything else (physical phones, other emulators, a renamed AVD)
# even if config.yaml asks. Edit this constant if you genuinely need to point
# the daemon at a different AVD — it's a deliberate, code-level change.
LOCKED_AVD = "CODM_Pixel9"


def _emulator_binary() -> Path:
    """Find Android Studio's emulator binary. Prefers the canonical SDK path,
    falls back to the one on PATH. Errors out clearly if not found — that's
    a real misconfiguration, not something to silently ignore."""
    candidates = [
        Path.home() / "Library/Android/sdk/emulator/emulator",
        Path(os.environ.get("ANDROID_HOME", "")) / "emulator/emulator",
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    found = subprocess.run(["which", "emulator"], capture_output=True, text=True).stdout.strip()
    if found:
        return Path(found)
    raise RuntimeError(
        "Couldn't locate the `emulator` binary. Install Android Studio's emulator "
        "package or set ANDROID_HOME so emulator/emulator is reachable."
    )


def ensure_avd_running(
    avd_name: str,
    boot_timeout: float = 240.0,
    post_boot_settle_seconds: float = 10.0,
) -> str:
    """If the locked AVD is already attached via ADB, return its serial.
    Otherwise launch it via the Android emulator binary, wait for it to come
    online + finish booting + settle, and return the serial.

    Refuses to start anything other than `LOCKED_AVD` — a guard against a
    caller (or a future config typo) trying to fire up an unrelated AVD.

    The post_boot_settle handles a real-world gotcha: `sys.boot_completed=1`
    means Android's init has finished, but Play Services, the launcher,
    storage mounts, and PackageManager often need another ~30s before apps
    can be launched cleanly. We only sleep this delay if we actually booted
    the AVD; warm reuse skips it entirely.
    """
    import logging as _log
    log = _log.getLogger(__name__)

    if avd_name != LOCKED_AVD:
        raise RuntimeError(
            f"Refusing to launch AVD {avd_name!r}: hard-locked to {LOCKED_AVD!r}."
        )
    # Warm path: already running -> return immediately, no settle.
    for d in AdbDevice.list_devices():
        if d.startswith("emulator-") and AdbDevice.emulator_avd_name(d) == avd_name:
            return d
    # Cold path: launch the emulator. Detach stdio so the child survives our
    # exit — the daemon doesn't want to babysit it.
    log.info("Launching AVD %r (cold)", avd_name)
    emu = _emulator_binary()
    subprocess.Popen(
        [
            str(emu), "-avd", avd_name,
            "-no-snapshot",
            "-gpu", "auto",
            "-netspeed", "full",
            "-netdelay", "none",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.time() + boot_timeout
    serial: str | None = None
    while time.time() < deadline:
        time.sleep(4.0)
        for d in AdbDevice.list_devices():
            if d.startswith("emulator-") and AdbDevice.emulator_avd_name(d) == avd_name:
                serial = d
                break
        if serial:
            break
    if not serial:
        raise RuntimeError(f"AVD {avd_name!r} didn't attach via ADB within {boot_timeout:.0f}s")
    log.info("AVD %r attached as %s — waiting for sys.boot_completed=1", avd_name, serial)
    # Wait for sys.boot_completed=1
    while time.time() < deadline:
        out = subprocess.run(
            ["adb", "-s", serial, "shell", "getprop", "sys.boot_completed"],
            check=False, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if out == "1":
            break
        time.sleep(3.0)
    else:
        raise RuntimeError(f"AVD {avd_name!r} attached but didn't finish booting within {boot_timeout:.0f}s")
    # Post-boot settle: Android's init is done but PackageManager / launcher /
    # Play Services often need another ~30s before app launches are reliable.
    if post_boot_settle_seconds > 0:
        log.info("AVD %r booted — settling %.0fs for services to initialize", avd_name, post_boot_settle_seconds)
        time.sleep(post_boot_settle_seconds)
    return serial


@dataclass
class AdbDevice:
    serial: str | None = None

    def _cmd(self, *args: str) -> list[str]:
        base = ["adb"]
        if self.serial:
            base += ["-s", self.serial]
        return base + list(args)

    def _run(self, *args: str, check: bool = True, capture: bool = True, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            self._cmd(*args),
            check=check,
            capture_output=capture,
            input=input_bytes,
        )

    # --- discovery -----------------------------------------------------------

    @staticmethod
    def list_devices() -> list[str]:
        out = subprocess.run(["adb", "devices"], check=True, capture_output=True, text=True).stdout
        serials = []
        for line in out.splitlines()[1:]:
            line = line.strip()
            if not line or line.startswith("*"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                serials.append(parts[0])
        return serials

    @staticmethod
    def emulator_avd_name(serial: str) -> str | None:
        """Ask an emulator what its AVD name is. Returns None if the device
        isn't an emulator or the lookup fails."""
        if not serial.startswith("emulator-"):
            return None
        try:
            out = subprocess.run(
                ["adb", "-s", serial, "emu", "avd", "name"],
                check=False, capture_output=True, text=True, timeout=5,
            ).stdout
        except Exception:
            return None
        # `adb emu avd name` returns the AVD name on the first line, "OK" on the second.
        for line in out.splitlines():
            line = line.strip()
            if line and line != "OK":
                return line
        return None

    @classmethod
    def auto(cls, preferred: str | None = None, target_avd: str | None = None) -> "AdbDevice":
        """Pick the locked CODM AVD. Refuses to attach to anything else.

        `preferred` and `target_avd` arguments are IGNORED for safety — they
        used to let the caller redirect to a different device, but that's a
        foot-gun (typos / leftover other emulators / physical phones in
        debug mode would silently get driven). The only AVD this project
        ever talks to is `LOCKED_AVD`. To change targets, edit the constant.
        """
        if preferred and preferred != "":
            log_msg = f"Ignoring adb_serial={preferred!r} — daemon is locked to AVD {LOCKED_AVD!r}."
            # Don't raise — just warn loudly so old configs still work.
            import logging as _log
            _log.getLogger(__name__).warning(log_msg)
        if target_avd and target_avd != LOCKED_AVD:
            raise RuntimeError(
                f"Refusing to attach: config target_avd={target_avd!r} but this project "
                f"is hard-locked to {LOCKED_AVD!r} (see lib/adb.py LOCKED_AVD). "
                "Either restore the config or edit the constant deliberately."
            )
        devices = cls.list_devices()
        if not devices:
            raise RuntimeError(
                f"No ADB devices found. Launch the locked AVD with: "
                f"emulator -avd {LOCKED_AVD}"
            )
        matches = [d for d in devices if cls.emulator_avd_name(d) == LOCKED_AVD]
        if len(matches) == 1:
            return cls(serial=matches[0])
        if not matches:
            attached_avds = {d: cls.emulator_avd_name(d) for d in devices if d.startswith("emulator-")}
            raise RuntimeError(
                f"Locked AVD {LOCKED_AVD!r} not attached. Attached: {attached_avds}. "
                f"Launch with: emulator -avd {LOCKED_AVD}"
            )
        raise RuntimeError(
            f"Multiple emulators claim to be {LOCKED_AVD!r}: {matches}. "
            "Kill the duplicates with `adb -s <serial> emu kill`."
        )

    # --- input ---------------------------------------------------------------

    def tap(self, x: int, y: int) -> None:
        self._run("shell", "input", "tap", str(x), str(y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self._run("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms))

    def key(self, keycode: str) -> None:
        self._run("shell", "input", "keyevent", keycode)

    def back(self) -> None:
        self.key("KEYCODE_BACK")

    def home(self) -> None:
        self.key("KEYCODE_HOME")

    def wake(self) -> None:
        # KEYCODE_WAKEUP (224) turns the screen on without toggling it back off
        # if it was already on (unlike KEYCODE_POWER, which is a toggle).
        self.key("KEYCODE_WAKEUP")

    # --- lock screen ---------------------------------------------------------

    def is_locked(self) -> bool:
        """Best-effort keyguard check via `dumpsys window`. Returns False when
        the state can't be determined so we never draw a pattern gesture onto
        an already-unlocked home screen (which could tap/drag real UI)."""
        out = subprocess.run(
            self._cmd("shell", "dumpsys", "window"),
            check=False, capture_output=True, text=True,
        ).stdout
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("mDreamingLockscreen"):
                return s.endswith("true")
        # Fallback: keyguard owns the focused window while locked.
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("mCurrentFocus") and ("Keyguard" in s or "NotificationShade" in s):
                return True
        return False

    @staticmethod
    def _dot_xy(idx: int, origin: tuple[int, int], spacing: tuple[int, int]) -> tuple[int, int]:
        """Pixel center of dot `idx` on the 3x3 pattern grid. Dots are numbered
        like a phone keypad: 1=top-left, 2=top-mid, 3=top-right, ... 9=bottom-right."""
        col = (idx - 1) % 3
        row = (idx - 1) // 3
        return (origin[0] + col * spacing[0], origin[1] + row * spacing[1])

    def draw_pattern(self, pattern: list[int], origin, spacing) -> None:
        """Draw an unlock pattern as a single continuous gesture using
        `input motionevent` (DOWN on the first dot, MOVE through each subsequent
        dot, UP on the last). One interpolated midpoint is inserted between
        consecutive dots so the swept path reliably crosses each dot's hit
        region rather than teleporting between centers."""
        origin = (int(origin[0]), int(origin[1]))
        spacing = (int(spacing[0]), int(spacing[1]))
        pts = [self._dot_xy(i, origin, spacing) for i in pattern]
        if len(pts) < 2:
            return
        parts = [f"input motionevent DOWN {pts[0][0]} {pts[0][1]}"]
        prev = pts[0]
        for cur in pts[1:]:
            mid = ((prev[0] + cur[0]) // 2, (prev[1] + cur[1]) // 2)
            parts.append(f"input motionevent MOVE {mid[0]} {mid[1]}")
            parts.append(f"input motionevent MOVE {cur[0]} {cur[1]}")
            prev = cur
        parts.append(f"input motionevent UP {pts[-1][0]} {pts[-1][1]}")
        # Chain in one device shell so the events form a single uninterrupted
        # gesture (separate adb invocations add ~100ms gaps that can split it).
        self._run("shell", " ; ".join(parts))

    def keep_awake(self) -> None:
        """Stop the display sleeping during the claim flow. WAKEUP alone leaves
        the emulator in a dark always-on state where injected gestures don't
        register, and the default screen-off timeout is only a few seconds.
        Both settings persist in the AVD's userdata, so this is idempotent and
        cheap to re-apply each cycle (guards against a wiped/recreated AVD)."""
        # 7 = stay on while charging on AC | USB | wireless. The emulator always
        # reports charging, so this holds the screen on indefinitely.
        self._run("shell", "settings", "put", "global", "stay_on_while_plugged_in", "7", check=False)
        self._run("shell", "settings", "put", "system", "screen_off_timeout", "1800000", check=False)

    def ensure_unlocked(self, lock_cfg: dict, attempts: int = 2) -> bool:
        """Wake the screen and, if the keyguard is up, draw the configured
        pattern to unlock. No-op if already unlocked. Returns True if the
        device ends up unlocked.

        Sequence (matches what actually works on the Pixel 9 AVD, Android 15):
        WAKEUP -> tap (make the display interactive; WAKEUP alone can leave a
        dark AOD that ignores gestures) -> swipe up to reveal the pattern
        bouncer -> draw the pattern. The bouncer is a secure surface so it
        screenshots as black; we verify success via the keyguard flag, not a
        template match."""
        import logging as _log
        log = _log.getLogger(__name__)
        self.keep_awake()
        tw = lock_cfg.get("tap_wake_xy", [720, 1500])
        tw = (int(tw[0]), int(tw[1]))
        self.wake()
        time.sleep(0.5)
        # A tap fully wakes the display to the interactive lock screen.
        self.tap(*tw)
        time.sleep(0.5)
        if not self.is_locked():
            log.info("Screen already unlocked")
            return True
        pattern = lock_cfg.get("pattern") or []
        origin = lock_cfg.get("grid_origin")
        spacing = lock_cfg.get("grid_spacing")
        if not (pattern and origin and spacing):
            log.warning("Screen locked but lockscreen config incomplete "
                        "(pattern/grid_origin/grid_spacing) — cannot unlock")
            return False
        reveal = lock_cfg.get("reveal_swipe")
        settle = float(lock_cfg.get("settle_after_unlock_seconds", 2.0))
        for i in range(1, attempts + 1):
            self.wake()
            self.tap(*tw)
            time.sleep(0.5)
            if reveal:
                # Swipe up to reveal the pattern bouncer.
                self.swipe(*[int(v) for v in reveal])
                time.sleep(0.8)
            log.info("Drawing unlock pattern (attempt %d/%d)", i, attempts)
            self.draw_pattern(pattern, origin, spacing)
            time.sleep(settle)
            if not self.is_locked():
                log.info("Unlocked on attempt %d", i)
                return True
            log.warning("Still locked after attempt %d", i)
        return not self.is_locked()

    # --- app lifecycle -------------------------------------------------------

    def is_app_foreground(self, package: str) -> bool:
        # `dumpsys window` (no subcommand) emits a top-of-output line:
        #   mCurrentFocus=Window{... <package>/<activity>}
        # Grepping just that line is far more robust than scanning the full
        # `dumpsys window windows` dump, which on Android 15 doesn't reliably
        # include mCurrentFocus.
        out = subprocess.run(
            self._cmd("shell", "dumpsys", "window"),
            check=False, capture_output=True, text=True,
        ).stdout
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("mCurrentFocus") and f"{package}/" in line:
                return True
        return False

    def launch_app(self, package: str, activity: str | None = None) -> None:
        if activity:
            self._run("shell", "am", "start", "-n", f"{package}/{activity}")
        else:
            # monkey picks the default launcher activity; quieter than `am start --user 0`
            self._run("shell", "monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1")

    def force_stop(self, package: str) -> None:
        self._run("shell", "am", "force-stop", package)

    # --- screen --------------------------------------------------------------

    def screencap(self) -> np.ndarray:
        """Pull a PNG screenshot off the device and decode to a BGR ndarray.
        Uses `exec-out` to keep the binary stream clean (no \\r\\n translation)."""
        result = subprocess.run(
            self._cmd("exec-out", "screencap", "-p"),
            check=True, capture_output=True,
        )
        arr = np.frombuffer(result.stdout, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("screencap returned no decodable image — is the AVD still running?")
        return img

    def screencap_to(self, path: Path) -> Path:
        img = self.screencap()
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
        return path

    # --- waits ---------------------------------------------------------------

    def wait_app_foreground(self, package: str, timeout: float = 30.0, poll: float = 1.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_app_foreground(package):
                return True
            time.sleep(poll)
        return False
