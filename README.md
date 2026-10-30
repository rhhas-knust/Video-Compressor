# Video Compressor

A command-line tool that shrinks video files while keeping them looking the same.
It wraps [ffmpeg](https://ffmpeg.org) and uses constant-quality (CRF) encoding, so
bits are spent where the picture needs them instead of on a fixed bitrate.

## Requirements
- Python 3.8+
- `ffmpeg` and `ffprobe` on your PATH

## Usage
```bash
python3 videocompressor.py movie.mov                  # -> movie_compressed.mp4
python3 videocompressor.py clips/ -r -o out/          # whole folder, recursive
python3 videocompressor.py movie.mp4 -p high --verify # near-lossless + SSIM score
python3 videocompressor.py movie.mp4 --target-size 25 # aim for ~25 MB (two-pass)
python3 videocompressor.py movie.mp4 --max-height 1080
python3 videocompressor.py movie.mp4 -c h264          # max device compatibility
```

| Option | Meaning |
|---|---|
| `-p high\|balanced\|small` | quality preset (default `balanced`) |
| `-c h265\|h264` | codec; H.265 is ~30-50% smaller at equal quality |
| `--crf N` | manual quality (lower = better) |
| `--speed` | encoder effort; `slow` gives smaller files |
| `--max-height` | downscale only if taller than this |
| `--target-size MB` | two-pass encode to a target size |
| `--verify` | print SSIM (>=0.98 excellent, >=0.95 good) |
| `--dry-run` | show the ffmpeg command only |

Notes: audio is copied untouched when already AAC/MP3/Opus/Vorbis, otherwise
re-encoded to AAC 128k. Outputs that aren't smaller than the original are
discarded (override with `--keep-larger`). The original is never modified.

## Tests
```bash
pip install pytest && python3 -m pytest tests
```

## Web version (no install, works on Chromebook)
`web/index.html` is a single-file browser app. Open it in Chrome, pick a video,
choose High / Balanced / Smallest and save the result. Nothing is uploaded; the
video is re-encoded in the tab using the browser's built-in encoder. It runs in
real time (a 5-minute video takes about 5 minutes) and the tab must stay visible.
