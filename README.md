# codm-dmz-claimer

Continuously claim DMZ LST Hunt rewards on CODM by driving an Android Studio AVD
via ADB + OpenCV template matching. Designed to run as a background daemon on
the Mac that hosts the emulator.

## Heads up — ban risk

Automating taps inside the live CODM game is botting under Activision's ToS.
This project deliberately keeps the cadence slow (default every 3h, plus
jitter) to look like a player who returns intermittently, but it cannot make
detection impossible. If you also run the codashop freebie automation on the
same UID, a ban here would take that down too.

## How it works

1. **AVD + CODM** stay running on your Mac (you log in once, manually).
2. `daemon.py` wakes every ~3h, calls `claimer.py`, sleeps again.
3. `claimer.py` brings CODM to the foreground, walks the navigation path
   to the DMZ LST Hunt rewards screen, taps every visible CLAIM button,
   and exits.
4. Navigation = a list of `Step`s in `lib/flow.py`. Each step waits for a
   reference image (a "template") to appear, then taps it. You capture
   those reference images from your own emulator with `capture.py`.

Two of the rewards refresh every 3-4h and the bigger one every 8h. A 3h
loop guarantees every reward is claimed within ~1h of becoming available
with no missed cycles.

## One-time setup

### 1. Android Studio AVD

- Install Android Studio.
- Tools → Device Manager → Create Device. Use a phone profile (Pixel 5 / 6
  works fine) with **API 33+** and at least **4GB RAM, 8GB internal storage**.
  Hardware acceleration ("Hypervisor.framework" on Apple Silicon) on.
- Boot the AVD once and verify it shows up:
  ```bash
  adb devices
  ```
  You should see one entry like `emulator-5554  device`.

### 2. Install CODM

The Play Store inside a stock AVD works on x86_64 system images; on Apple
Silicon (arm64 system images) it sometimes won't list CODM. Fallbacks:

- Install via Play Store inside the AVD (preferred).
- Or sideload an APK: `adb install path/to/codm.apk`.

Log into your Activision account **once, by hand**, in the running AVD.
Complete any 2FA the game throws at you. From here on the daemon assumes
the app stays logged in (the AVD's data partition persists).

### 3. Python deps

```bash
cd ~/Documents/Khalid/codm-dmz-claimer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure

Open `config.yaml`. The defaults are reasonable; the ones you likely need
to change:

- `package`: Activision global = `com.activision.callofduty.shooter`. Garena
  SEA build = `com.garena.game.codm`. Find yours with:
  ```bash
  adb shell pm list packages | grep -i -E 'codm|callofduty'
  ```
- `adb_serial`: leave blank if you only ever have one AVD attached.

### 5. Capture templates

This is the part that's specific to your emulator's resolution. Each step
in `lib/flow.py:DEFAULT_STEPS` expects a PNG under `templates/`. Workflow
per step:

1. In the running AVD, navigate (manually) to the screen the step targets.
   For step 1 (`main_menu_marker`) that's the CODM lobby. For step 4
   (`claim_buttons`) it's the DMZ LST Hunt rewards page.
2. Grab a full screenshot:
   ```bash
   python capture.py screen --out logs/now.png
   ```
3. Open `logs/now.png` (Preview works). Hover over the button you want to
   match — Preview shows pixel coordinates in the title bar with View →
   Show Inspector. Note `(x1, y1)` top-left and `(x2, y2)` bottom-right.
4. Crop it to a template:
   ```bash
   python capture.py crop --src logs/now.png --rect X1 Y1 X2 Y2 \
       --out templates/04_claim_button.png
   ```
   Pick a tight crop — just the button, no surrounding UI. Tighter = fewer
   false matches.
5. Verify it matches with the AVD still on that screen:
   ```bash
   python capture.py match --template templates/04_claim_button.png
   # → center=(620,1240) score=0.94 size=240x60
   ```
   Anything ≥ 0.85 is solid. Re-crop if you're getting low scores.

Templates expected by the default flow:

| File | Captured from | Notes |
| --- | --- | --- |
| `01_main_menu_marker.png` | Lobby | Any always-on element (MULTIPLAYER tab, your operator name, the menu icon) |
| `02_event_hub_entry.png` | Lobby | Whatever button opens the events / DMZ hub |
| `03_dmz_lst_hunt_tab.png` | Event hub | The DMZ LST Hunt card or tab |
| `04_claim_button.png` | LST Hunt page | The CLAIM button itself — captured while it is in its "ready to claim" state, not its locked state |
| `05_ok_button.png` | After clicking CLAIM | The OK / Got it / Close popup that appears after a successful claim |
| `06_back_button.png` | LST Hunt page | The arrow / X to go back to the lobby |

If a step in your flow doesn't exist (e.g. there's no confirm popup, the
claim just dismisses itself), leave the template missing — `flow.py`
treats the step as skip-eligible.

### 6. Dry-run the flow

```bash
# AVD running, CODM at the lobby
python claimer.py --dry-run --verbose
```

It will log every tap it *would* make. Read `logs/claimer.log` — each step
should report a match score. If a step never matches, recapture that
template tighter, or lower `match_threshold` in `config.yaml` (try 0.78).

When dry-run looks right:

```bash
python claimer.py            # one real attempt
cat logs/<latest>_summary.json
```

### 7. Start the daemon

Foreground for testing:
```bash
python daemon.py
```

Background, persistent (recommended) — install the launchd agent:
```bash
cp com.codm-dmz-claimer.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.codm-dmz-claimer.plist
```

To stop:
```bash
launchctl unload ~/Library/LaunchAgents/com.codm-dmz-claimer.plist
```

Logs:
- `logs/claimer.log` — structured per-cycle log
- `logs/<timestamp>_summary.json` — machine-readable result of each claim
- `logs/<timestamp>_final.png` — screenshot at the end of each cycle
- `logs/launchd.{out,err}.log` — stdout/stderr from the launchd agent

## Tuning

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Step matches in dry-run but taps land on wrong spot | Template crop included too much background | Re-crop tighter around the button itself |
| Score consistently 0.70-0.80 | AVD resolution drifted, or template captured before a UI animation finished | Re-capture; multi-scale matching already tries ±10% |
| Skips `claim_buttons` every time | Buttons are in their "on cooldown" state (greyed, timer shown) — that's expected most of the time | None — the daemon will catch them next cycle when they refresh |
| Aborts at `main_menu_marker` | CODM crashed / showed an update prompt / "are you there?" idle popup | Add a step for the popup, or use `--verbose` and inspect `logs/<ts>_final.png` |
| Daemon claims nothing for many cycles | Templates drifted after a game patch | Recapture; CODM updates UI a few times a year |

## File map

```
codm-dmz-claimer/
├── claimer.py           # one-shot attempt
├── daemon.py            # loop forever
├── capture.py           # screenshot + crop + verify helper
├── config.yaml          # all knobs
├── com.codm-dmz-claimer.plist   # launchd agent
├── lib/
│   ├── adb.py           # ADB wrapper
│   ├── vision.py        # OpenCV template matching
│   └── flow.py          # step list for the claim sequence
├── templates/           # PNGs you capture (per-resolution, not portable)
├── logs/                # screenshots, summaries, claimer.log
└── requirements.txt
```
