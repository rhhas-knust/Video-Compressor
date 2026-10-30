import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import videocompressor as vc


def make_video(path, codec="mpeg4"):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=640x360:rate=24:duration=4", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=4", "-c:v", codec, "-b:v", "4M",
                    "-c:a", "aac", str(path)], check=True)


def test_compress_reduces_size_and_keeps_quality(tmp_path):
    src = tmp_path / "in.mp4"
    make_video(src)
    rc = vc.main([str(src), "-o", str(tmp_path / "out"), "--speed", "ultrafast", "--verify"])
    out = tmp_path / "out" / "in_compressed.mp4"
    assert rc == 0 and out.exists()
    assert out.stat().st_size < src.stat().st_size
    assert vc.ssim(src, out) > 0.95


def test_target_size(tmp_path):
    src = tmp_path / "in.mp4"
    make_video(src)
    vc.main([str(src), "-o", str(tmp_path / "o"), "--target-size", "0.4", "--speed", "ultrafast"])
    out = tmp_path / "o" / "in_compressed.mp4"
    assert out.exists() and out.stat().st_size < 0.6 * 1024 * 1024


def test_dry_run_creates_nothing(tmp_path):
    src = tmp_path / "in.mp4"
    make_video(src)
    vc.main([str(src), "--dry-run"])
    assert not (tmp_path / "in_compressed.mp4").exists()
