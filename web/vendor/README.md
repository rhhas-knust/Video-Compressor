Vendored, unmodified third-party builds:

- `mp4box.all.min.js` — [mp4box.js](https://github.com/gpac/mp4box.js) 0.5.2, BSD-3-Clause. Reads MP4/MOV containers.
- `mp4-muxer.js` — [mp4-muxer](https://github.com/Vanilagy/mp4-muxer) 5.1.3, MIT. Writes MP4 containers.
- `ffmpeg/`, `ffmpeg-util/` — [ffmpeg.wasm](https://github.com/ffmpegwasm/ffmpeg.wasm) wrapper 0.12.15 and util 0.12.2 (ESM builds), MIT. Hosted here so its worker is same-origin; converts camcorder sound.

Not vendored: transcription loads [transformers.js](https://github.com/huggingface/transformers.js) 4.3.0 (Apache-2.0) and its ONNX Runtime Web build from jsDelivr at runtime, and camcorder conversion loads the ffmpeg.wasm core 0.12.10 (GPL-2.0-or-later build of FFmpeg) from jsDelivr when an MTS file's sound is converted.
