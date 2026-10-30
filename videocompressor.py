#!/usr/bin/env python3
"""Video Compressor: shrink videos with ffmpeg while preserving visual quality.

Uses constant-quality (CRF) encoding with H.265/H.264, copies audio when it is
already efficient, and can verify the result with an SSIM quality score.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".wmv", ".ts"}

# CRF per preset and codec. Lower CRF = higher quality / bigger file.
PRESETS = {
    "high":     {"libx265": 20, "libx264": 18},   # visually lossless
    "balanced": {"libx265": 24, "libx264": 22},   # best size/quality trade-off
    "small":    {"libx265": 28, "libx264": 26},   # smallest, minor loss
}
COPYABLE_AUDIO = {"aac", "mp3", "opus", "vorbis"}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def probe(path):
    r = run(["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)])
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {r.stderr.strip()}")
    return json.loads(r.stdout)


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def build_command(src, dst, info, codec, crf, speed, max_height, target_kbps=None,
                  passno=None, passlog=None):
    streams = info.get("streams", [])
    audio = next((s for s in streams if s["codec_type"] == "audio"), None)
    video = next((s for s in streams if s["codec_type"] == "video"), None)
    if video is None:
        raise RuntimeError("no video stream found")

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
           "-i", str(src), "-map", "0:v:0", "-map", "0:a?", "-map", "0:s?"]
    cmd += ["-c:v", codec, "-preset", speed, "-pix_fmt", "yuv420p"]
    if target_kbps:
        cmd += ["-b:v", f"{target_kbps}k"]
    else:
        cmd += ["-crf", str(crf)]
    if codec == "libx265":
        cmd += ["-tag:v", "hvc1"]  # Apple/QuickTime compatibility
    if max_height and int(video.get("height", 0)) > max_height:
        cmd += ["-vf", f"scale=-2:{max_height}"]

    if passno:
        cmd += ["-pass", str(passno), "-passlogfile", passlog]
        if passno == 1:
            return cmd + ["-an", "-sn", "-f", "null", "-"]

    if audio and audio.get("codec_name") in COPYABLE_AUDIO:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    cmd += ["-c:s", "mov_text" if dst.suffix.lower() in (".mp4", ".m4v", ".mov") else "copy"]
    cmd += ["-movflags", "+faststart", str(dst)]
    return cmd


def ssim(original, compressed):
    """Return average SSIM (1.0 = identical), scaling to match if needed."""
    r = run(["ffmpeg", "-hide_banner", "-i", str(compressed), "-i", str(original),
             "-lavfi", "[0:v][1:v]scale2ref=flags=bicubic[a][b];[a][b]ssim",
             "-f", "null", "-"])
    m = re.search(r"All:([0-9.]+)", r.stderr)
    return float(m.group(1)) if m else None


def compress(src, dst, args):
    info = probe(src)
    duration = float(info["format"].get("duration", 0) or 0)
    codec = "libx265" if args.codec == "h265" else "libx264"
    crf = args.crf if args.crf is not None else PRESETS[args.preset][codec]
    orig_size = src.stat().st_size

    target_kbps = None
    if args.target_size:
        if duration <= 0:
            raise RuntimeError("cannot use --target-size: unknown duration")
        total_kbps = args.target_size * 8 * 1024 / duration
        target_kbps = max(100, int(total_kbps - 128))  # leave room for audio

    print(f"\n{src.name}: {human(orig_size)}, {duration:.1f}s -> {dst.name} "
          f"({codec}, {'~%dk' % target_kbps if target_kbps else 'CRF %d' % crf})")
    if args.dry_run:
        print(" ", " ".join(build_command(src, dst, info, codec, crf, args.speed,
                                          args.max_height, target_kbps)))
        return None

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part" + dst.suffix)
    try:
        if target_kbps:
            log = str(dst.with_suffix(".passlog"))
            for p in (1, 2):
                cmd = build_command(src, tmp, info, codec, crf, args.speed,
                                    args.max_height, target_kbps, p, log)
                if subprocess.run(cmd).returncode != 0:
                    raise RuntimeError("ffmpeg failed")
            for f in dst.parent.glob(dst.stem + ".passlog*"):
                f.unlink()
        else:
            cmd = build_command(src, tmp, info, codec, crf, args.speed, args.max_height)
            if subprocess.run(cmd).returncode != 0:
                raise RuntimeError("ffmpeg failed")
        new_size = tmp.stat().st_size
        if new_size >= orig_size and not args.keep_larger:
            tmp.unlink()
            print(f"  skipped: result ({human(new_size)}) is not smaller than the original")
            return None
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink()

    saved = 100 * (1 - new_size / orig_size)
    msg = f"  done: {human(orig_size)} -> {human(new_size)} (saved {saved:.1f}%)"
    if args.verify:
        s = ssim(src, dst)
        if s is not None:
            verdict = "excellent" if s >= 0.98 else "good" if s >= 0.95 else "noticeable loss"
            msg += f", SSIM {s:.4f} ({verdict})"
    print(msg)
    return orig_size, new_size


def collect(inputs, recursive):
    files = []
    for p in inputs:
        p = Path(p)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            files += sorted(f for f in it if f.suffix.lower() in VIDEO_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: {p} not found", file=sys.stderr)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reduce video size while keeping quality.")
    ap.add_argument("inputs", nargs="+", help="video files or folders")
    ap.add_argument("-o", "--output-dir", help="output folder (default: next to input)")
    ap.add_argument("-p", "--preset", choices=PRESETS, default="balanced",
                    help="quality preset (default: balanced)")
    ap.add_argument("-c", "--codec", choices=["h265", "h264"], default="h265",
                    help="h265 = smaller files, h264 = most compatible (default: h265)")
    ap.add_argument("--crf", type=int, help="override CRF (lower = better quality)")
    ap.add_argument("--speed", default="medium",
                    choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                             "medium", "slow", "slower", "veryslow"],
                    help="encoder effort; slower = smaller file (default: medium)")
    ap.add_argument("--max-height", type=int, help="downscale if taller than this (e.g. 1080)")
    ap.add_argument("--target-size", type=float, metavar="MB",
                    help="aim for this file size using two-pass encoding")
    ap.add_argument("--verify", action="store_true", help="report SSIM quality score")
    ap.add_argument("-r", "--recursive", action="store_true", help="recurse into folders")
    ap.add_argument("--keep-larger", action="store_true",
                    help="keep output even if it is not smaller")
    ap.add_argument("--dry-run", action="store_true", help="print ffmpeg command only")
    args = ap.parse_args(argv)

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"error: {tool} not found; install ffmpeg first")

    files = collect(args.inputs, args.recursive)
    if not files:
        sys.exit("error: no video files found")

    total_in = total_out = failed = 0
    for src in files:
        out_dir = Path(args.output_dir) if args.output_dir else src.parent
        dst = out_dir / f"{src.stem}_compressed.mp4"
        if dst.resolve() == src.resolve():
            continue
        try:
            res = compress(src, dst, args)
        except Exception as e:
            failed += 1
            print(f"  error: {e}", file=sys.stderr)
            continue
        if res:
            total_in += res[0]
            total_out += res[1]

    if total_in:
        print(f"\nTotal: {human(total_in)} -> {human(total_out)} "
              f"(saved {100 * (1 - total_out / total_in):.1f}%)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
