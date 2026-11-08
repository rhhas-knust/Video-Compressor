"""Runs the JavaScript tests for the MTS (camcorder) reader in web/mts.js."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
node = shutil.which("node")
ffmpeg = shutil.which("ffmpeg")


@pytest.mark.skipif(not node, reason="needs node")
def test_mts_unit_tests():
    r = subprocess.run([node, str(HERE / "mts_test.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(not node or not ffmpeg, reason="needs node and ffmpeg")
def test_demuxes_interlaced_camcorder_style_file(tmp_path):
    src = tmp_path / "cam.MTS"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=25:duration=3",
                    "-f", "lavfi", "-i", "sine=duration=3:sample_rate=48000", "-c:v", "libx264", "-preset", "ultrafast",
                    "-flags", "+ildct+ilme", "-c:a", "ac3", "-f", "mpegts", "-mpegts_m2ts_mode", "1", str(src)], check=True)
    r = subprocess.run([node, str(HERE / "mts_test.js"), str(src)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    info = json.loads(next(l for l in r.stdout.splitlines() if l.startswith("{")))
    assert info == {"frames": 75, "keys": info["keys"], "width": 1920, "height": 1080, "interlaced": True}
    assert info["keys"] >= 1
