"""Capture real dashboard screenshots via headless Chrome + DevTools protocol.

Usage: start the dashboard (`make dashboard`), then
  python scripts/capture_screenshots.py [width] [frames]
`frames` writes viewport-size frames to /tmp/sfrl-frames for the demo GIF.
Writes docs/screenshots/*.png. Waits until each page reports its title, so a loading
skeleton is never saved.
"""

import asyncio
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from websockets.asyncio.client import connect

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PAGES = {
    "01_digital_twin": ("Factory Digital Twin", "Factory Digital Twin"),
    "02_policy_comparison": ("Policy Comparison", "Scheduling Policy Comparison"),
    "03_machine_analytics": ("Machine Analytics", "Machine Analytics"),
    "04_experiment_tracking": ("Experiment Tracking", "Experiment Tracking"),
    "05_robustness": ("Robustness", "Robustness Analysis"),
    "06_schedule_gantt": ("Schedule Visualization", "Schedule Visualization"),
}
MAIN_BOTTOM_JS = (
    "Math.ceil(document.querySelector('[data-testid=stMainBlockContainer]')"
    ".getBoundingClientRect().bottom) + 60"
)
FRAME_DIR = Path("/tmp/sfrl-frames")  # viewport-height frames for docs/demo.gif
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


async def main(width: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            "--remote-debugging-port=9333",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={width},900",
            "--user-data-dir=/tmp/sfrl-chrome",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen("http://localhost:9333/json"))
                break
            except OSError:
                time.sleep(0.2)
        ws = await connect(
            [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"], max_size=None
        )
        n = 0

        async def call(method, **params):
            nonlocal n
            n += 1
            await ws.send(json.dumps({"id": n, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == n:
                    return msg.get("result", {})

        await call(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=900,
            deviceScaleFactor=1,
            mobile=width < 600,
        )
        for name, (page, marker) in PAGES.items():
            await call(
                "Page.navigate", url=f"http://localhost:8501/?page={page.replace(' ', '%20')}"
            )
            for _ in range(60):
                await asyncio.sleep(1)
                r = await call(
                    "Runtime.evaluate",
                    returnByValue=True,
                    expression=("(document.querySelector('h1')||{}).innerText||''"),
                )
                title = r["result"].get("value", "")
                busy = await call(
                    "Runtime.evaluate",
                    returnByValue=True,
                    expression=("!!document.querySelector('[data-testid=stStatusWidget]')"),
                )
                if marker in title and not busy["result"].get("value"):
                    break
            await asyncio.sleep(2)
            h = await call(
                "Runtime.evaluate",
                returnByValue=True,
                expression=MAIN_BOTTOM_JS,
            )
            height = 800 if FRAMES else max(700, min(int(h["result"]["value"]), 4000))
            await call(
                "Emulation.setDeviceMetricsOverride",
                width=width,
                height=height,
                deviceScaleFactor=1,
                mobile=width < 600,
            )
            shot = await call("Page.captureScreenshot", format="png")
            suffix = "" if width >= 1000 else f"_w{width}"
            target = FRAME_DIR if FRAMES else OUT
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{name}{suffix}.png").write_bytes(base64.b64decode(shot["data"]))
            print("saved", name, "title:", title, "height", height)
    finally:
        proc.terminate()


FRAMES = len(sys.argv) > 2 and sys.argv[2] == "frames"
asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1400))
