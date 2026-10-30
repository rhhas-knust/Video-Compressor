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
`web/index.html` is a browser app (open it in Chrome). Pick a video, choose
High / Balanced / Smallest and save the result. Nothing is uploaded.

- **Fast mode** (MP4/MOV): reads the file directly with WebCodecs and re-encodes
  the picture with the device's encoder, usually several times faster than
  playback. Audio is copied unchanged. "Smallest file" uses H.265 or AV1 when
  available; AV1 shrinks already-compressed videos that H.264 cannot.
  "Plays everywhere" uses H.264. If the output is on course to be no smaller
  than the original, it stops early and says so.
- **Real-time mode** (other formats, e.g. MKV): plays the video through once
  with MediaRecorder, so it takes as long as the video.

Hosted with GitHub Pages (Settings → Pages → Deploy from branch `main`, `/ (root)`):
https://rhhas-knust.github.io/Video-Compressor/ redirects to the app in `web/`.

`web/vendor/` holds the two libraries it uses (mp4box.js, mp4-muxer); serve the
`web/` folder over HTTPS or localhost, since WebCodecs needs a secure context.
