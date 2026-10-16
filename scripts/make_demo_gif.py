"""Assemble /tmp/sfrl-frames/*.png (`capture_screenshots.py 1400 frames`) into docs/demo.gif."""

from pathlib import Path

from PIL import Image

frames = sorted(Path("/tmp/sfrl-frames").glob("0*.png"))
imgs = [Image.open(f).convert("RGB").resize((900, 514), Image.Resampling.LANCZOS) for f in frames]
imgs = [i.quantize(colors=96, method=Image.Quantize.MEDIANCUT) for i in imgs]
out = Path(__file__).resolve().parents[1] / "docs" / "demo.gif"
imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=2200, loop=0, optimize=True)
print(out, len(imgs), "frames,", out.stat().st_size // 1024, "KB")
