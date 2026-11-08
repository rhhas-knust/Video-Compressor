// Unit tests for web/mts.js (run by tests/test_mts.py, or: node tests/mts_test.js [sample.mts])
const assert = require('assert');
const path = require('path');
const MTS = require(path.join(__dirname, '..', 'web', 'mts.js'));

// Writes H.264 syntax elements and adds emulation prevention, to build test NAL units.
function bitWriter() {
  const bits = [];
  return {
    u(n, v) { for (let i = n - 1; i >= 0; i--) bits.push((v >> i) & 1); return this; },
    ue(v) { const x = v + 1, n = Math.floor(Math.log2(x)); this.u(n, 0); return this.u(n + 1, x); },
    nal(header) {
      bits.push(1); while (bits.length % 8) bits.push(0);
      const raw = [header];
      for (let i = 0; i < bits.length; i += 8) raw.push(parseInt(bits.slice(i, i + 8).join(''), 2));
      const out = []; let zeros = 0;
      raw.forEach((b, i) => { if (i > 0 && zeros >= 2 && b <= 3) { out.push(3); zeros = 0; } out.push(b); zeros = b === 0 ? zeros + 1 : 0; });
      return Uint8Array.from(out);
    },
  };
}
// Main profile 1920x1080 SPS that allows field pictures (frame_mbs_only_flag = 0, no MBAFF)
const SPS = bitWriter().u(8, 77).u(8, 0).u(8, 40).ue(0).ue(0).ue(0).ue(0).ue(1).u(1, 0)
  .ue(119).ue(33).u(1, 0).u(1, 0).u(1, 1).u(1, 1).ue(0).ue(0).ue(0).ue(2).u(1, 0).nal(0x67);
const PPS = Uint8Array.from([0x68, 0xce, 0x3c, 0x80]);
const AUD = Uint8Array.from([0x09, 0xf0]);
function slice(idr, sliceType, frameNum, field, bottom) {
  const w = bitWriter().ue(0).ue(sliceType).ue(0).u(4, frameNum).u(1, field);
  if (field) w.u(1, bottom);
  return w.ue(0).nal(idr ? 0x65 : 0x41);
}

const tests = {
  'parses an interlaced SPS': () => {
    const s = MTS.parseSps(SPS);
    assert.strictEqual(s.width, 1920); assert.strictEqual(s.height, 1080);
    assert.strictEqual(s.interlaced, true); assert.strictEqual(s.log2MaxFrameNum, 4);
    assert.strictEqual(MTS.codecString(s), 'avc1.4d0028');
  },
  'joins the two fields of a frame into one sample': () => {
    const out = [], p = new MTS.FramePairer(s => out.push(s));
    p.push({ nals: [AUD, SPS, PPS, slice(true, 7, 0, 1, 0)], pts: 1000, dts: 900 });  // IDR top field
    p.push({ nals: [AUD, slice(false, 5, 0, 1, 1)], pts: 2800, dts: 2700 });          // its bottom field
    p.push({ nals: [AUD, slice(false, 5, 1, 1, 0)], pts: 4600, dts: 4500 });          // next frame, top
    p.push({ nals: [AUD, slice(false, 5, 1, 1, 1)], pts: 6400, dts: 6300 });          // next frame, bottom
    p.end();
    assert.strictEqual(out.length, 2);
    assert.deepStrictEqual(out.map(s => [s.pts, s.dts, s.key]), [[1000, 900, true], [4600, 4500, false]]);
    assert.strictEqual(out[0].nals.length, 6);
  },
  'keeps a lone field and frame pictures as they are': () => {
    const out = [], p = new MTS.FramePairer(s => out.push(s));
    p.push({ nals: [SPS, PPS, slice(true, 7, 0, 1, 0)], pts: 0, dts: 0 });
    p.push({ nals: [slice(false, 5, 1, 0, 0)], pts: 3600, dts: 3600 });  // frame picture: no pair
    p.push({ nals: [slice(false, 5, 2, 1, 1)], pts: 7200, dts: 7200 });  // bottom field, partner missing
    p.end();
    assert.deepStrictEqual(out.map(s => s.pts), [0, 3600, 7200]);
  },
  'writes length-prefixed NALs without AUDs': () => {
    const b = MTS.toLengthPrefixed([AUD, PPS]);
    assert.deepStrictEqual(Array.from(b), [0, 0, 0, 4, 0x68, 0xce, 0x3c, 0x80]);
  },
  'detects transport stream packet sizes': () => {
    const m = (size, off) => { const b = new Uint8Array(size * 3 + off); for (let k = 0; k < 3; k++) b[off + k * size] = 0x47; return b; };
    assert.strictEqual(MTS.packetSize(m(188, 0)), 188);
    assert.strictEqual(MTS.packetSize(m(192, 4)), 192);
    assert.strictEqual(MTS.packetSize(new Uint8Array(1000)), 0);
  },
};

// Optional: demux a real file and check every frame is accounted for.
const sample = process.argv[2];
if (sample) {
  tests['demuxes ' + path.basename(sample)] = () => {
    const buf = new Uint8Array(require('fs').readFileSync(sample));
    const size = MTS.packetSize(buf.subarray(0, 1024));
    assert.ok(size, 'not a transport stream');
    let n = 0, keys = 0, sps = null;
    const p = new MTS.FramePairer(s => { n++; if (s.key) keys++; sps = s.sps; });
    const d = new MTS.Demuxer(size, { onAccessUnit: au => p.push(au) });
    for (let o = 0; o < buf.length; o += 65536) d.push(buf.subarray(o, o + 65536));
    d.end(); p.end();
    console.log(JSON.stringify({ frames: n, keys, width: sps.width, height: sps.height, interlaced: sps.interlaced }));
  };
}

let failed = 0;
for (const [name, fn] of Object.entries(tests)) {
  try { fn(); console.log('ok   ' + name); } catch (e) { failed++; console.log('FAIL ' + name + '\n     ' + e.message); }
}
process.exit(failed ? 1 : 0);
