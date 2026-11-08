/*
 * Reads camcorder MTS/M2TS (MPEG transport stream, AVCHD) files and turns their H.264
 * video into MP4-ready samples, without re-encoding.
 *
 *   MTS.packetSize(bytes)           188, 192 (M2TS) or 204 if bytes start a transport stream, else 0
 *   new MTS.Demuxer(packetSize, cb) push(chunk) / end(); calls back
 *       cb.onStreams([{pid, type, kind}])      once the program table is read
 *       cb.onAccessUnit({nals, pts, dts})      each H.264 access unit (a frame, or one field)
 *       cb.onAudioPts(pid, pts)                the first timestamp seen on each audio stream
 *   new MTS.FramePairer(onSample)   push(accessUnit) / end(); joins the two fields of an
 *       interlaced (field-coded) frame into one sample, as MP4 requires:
 *       onSample({nals, pts, dts, key, sps})
 *   MTS.parseSps(nal), MTS.avcC(sps, pps), MTS.toLengthPrefixed(nals), MTS.codecString(sps)
 *
 * Timestamps are in 90 kHz ticks, unwrapped past the 33-bit rollover.
 */
(function (root) {
  'use strict';

  var STREAM_KINDS = {
    0x1b: 'h264', 0x24: 'hevc', 0x02: 'mpeg2video', 0x01: 'mpeg1video',
    0x81: 'ac3', 0x84: 'eac3', 0x87: 'eac3', 0x0f: 'aac', 0x11: 'aac-latm', 0x03: 'mp2', 0x04: 'mp2', 0x80: 'lpcm'
  };
  var AUDIO = { ac3: 1, eac3: 1, aac: 1, 'aac-latm': 1, mp2: 1, lpcm: 1 };

  function packetSize(b) {
    var sizes = [[188, 0], [192, 4], [204, 0]];
    for (var i = 0; i < sizes.length; i++) {
      var size = sizes[i][0], off = sizes[i][1], ok = b.length >= off + size * 3;
      for (var k = 0; ok && k < 3; k++) if (b[off + k * size] !== 0x47) ok = false;
      if (ok) return size;
    }
    return 0;
  }

  /* ---------- bit reading ---------- */

  // Removes emulation-prevention bytes (00 00 03 -> 00 00) so syntax elements can be read.
  function unescape(nal, maxBytes) {
    var n = Math.min(nal.length, maxBytes || nal.length), out = new Uint8Array(n), o = 0, zeros = 0;
    for (var i = 0; i < n; i++) {
      var v = nal[i];
      if (zeros >= 2 && v === 3) { zeros = 0; continue; }
      out[o++] = v;
      zeros = v === 0 ? zeros + 1 : 0;
    }
    return out.subarray(0, o);
  }
  function Bits(bytes, startByte) {
    this.b = bytes; this.p = (startByte || 0) * 8;
  }
  Bits.prototype.u = function (n) {
    var v = 0;
    for (var i = 0; i < n; i++) {
      var byte = this.b[this.p >> 3];
      if (byte === undefined) throw new Error('read past end of NAL');
      v = v * 2 + ((byte >> (7 - (this.p & 7))) & 1);
      this.p++;
    }
    return v;
  };
  Bits.prototype.ue = function () {
    var zeros = 0;
    while (this.u(1) === 0) { if (++zeros > 31) throw new Error('bad exp-Golomb code'); }
    return zeros ? Math.pow(2, zeros) - 1 + this.u(zeros) : 0;
  };
  Bits.prototype.se = function () {
    var k = this.ue();
    return k & 1 ? (k + 1) / 2 : -k / 2;
  };

  /* ---------- H.264 syntax ---------- */

  var HIGH_PROFILES = { 100: 1, 110: 1, 122: 1, 244: 1, 44: 1, 83: 1, 86: 1, 118: 1, 128: 1, 138: 1, 139: 1, 134: 1, 135: 1 };

  function skipScalingList(r, size) {
    var last = 8, next = 8;
    for (var j = 0; j < size; j++) {
      if (next !== 0) next = (last + r.se() + 256) % 256;
      last = next === 0 ? last : next;
    }
  }

  function parseSps(nal) {
    var b = unescape(nal, 256), r = new Bits(b, 1);
    var s = { nal: nal, profile: r.u(8), constraints: r.u(8), level: r.u(8) };
    s.id = r.ue();
    s.chromaFormat = 1; s.separateColourPlane = 0; s.bitDepthLuma = 8; s.bitDepthChroma = 8;
    if (HIGH_PROFILES[s.profile]) {
      s.chromaFormat = r.ue();
      if (s.chromaFormat === 3) s.separateColourPlane = r.u(1);
      s.bitDepthLuma = r.ue() + 8;
      s.bitDepthChroma = r.ue() + 8;
      r.u(1); // qpprime_y_zero_transform_bypass_flag
      if (r.u(1)) {
        for (var i = 0; i < (s.chromaFormat !== 3 ? 8 : 12); i++) if (r.u(1)) skipScalingList(r, i < 6 ? 16 : 64);
      }
    }
    s.log2MaxFrameNum = r.ue() + 4;
    s.pocType = r.ue();
    if (s.pocType === 0) s.log2MaxPocLsb = r.ue() + 4;
    else if (s.pocType === 1) {
      r.u(1); r.se(); r.se();
      var cycle = r.ue();
      for (var c = 0; c < cycle; c++) r.se();
    }
    s.maxRefFrames = r.ue();
    r.u(1); // gaps_in_frame_num_value_allowed_flag
    var wMbs = r.ue() + 1, hMapUnits = r.ue() + 1;
    s.frameMbsOnly = r.u(1);
    if (!s.frameMbsOnly) s.mbaff = r.u(1);
    r.u(1); // direct_8x8_inference_flag
    var cropL = 0, cropR = 0, cropT = 0, cropB = 0;
    if (r.u(1)) { cropL = r.ue(); cropR = r.ue(); cropT = r.ue(); cropB = r.ue(); }
    var subW = s.chromaFormat === 1 || s.chromaFormat === 2 ? 2 : 1, subH = s.chromaFormat === 1 ? 2 : 1;
    if (s.chromaFormat === 0 || s.separateColourPlane) { subW = 1; subH = 1; }
    var cropUnitX = subW, cropUnitY = subH * (2 - s.frameMbsOnly);
    s.codedWidth = wMbs * 16;
    s.codedHeight = hMapUnits * 16 * (2 - s.frameMbsOnly);
    s.width = s.codedWidth - (cropL + cropR) * cropUnitX;
    s.height = s.codedHeight - (cropT + cropB) * cropUnitY;
    s.interlaced = !s.frameMbsOnly;
    return s;
  }

  // Reads the start of a slice header: enough to find frame and field boundaries.
  function parseSliceHeader(nal, sps) {
    var r = new Bits(unescape(nal, 64), 1);
    var h = { type: nal[0] & 31, refIdc: (nal[0] >> 5) & 3 };
    h.firstMb = r.ue();
    h.sliceType = r.ue() % 5;
    h.ppsId = r.ue();
    if (!sps) return h;
    if (sps.separateColourPlane) r.u(2);
    h.frameNum = r.u(sps.log2MaxFrameNum);
    h.field = 0; h.bottom = 0;
    if (!sps.frameMbsOnly) {
      h.field = r.u(1);
      if (h.field) h.bottom = r.u(1);
    }
    return h;
  }
  function firstMbInSlice(nal) {
    try { return new Bits(unescape(nal, 16), 1).ue(); } catch (e) { return 0; }
  }
  function seiHasRecoveryPoint(nal) {
    var b = unescape(nal, 32), i = 1, type = 0;
    while (i < b.length && b[i] === 0xff) { type += 255; i++; }
    return i < b.length && type + b[i] === 6;
  }

  function splitNals(bytes) {
    var nals = [], i = 0, start = -1, n = bytes.length;
    while (i + 2 < n) {
      if (bytes[i] === 0 && bytes[i + 1] === 0 && bytes[i + 2] === 1) {
        if (start >= 0) nals.push(trimZeros(bytes, start, i));
        i += 3; start = i;
      } else i++;
    }
    if (start >= 0 && start < n) nals.push(trimZeros(bytes, start, n));
    return nals.filter(function (x) { return x.length > 0; });
  }
  function trimZeros(b, s, e) {
    while (e > s && b[e - 1] === 0) e--;
    return b.subarray(s, e);
  }

  function toLengthPrefixed(nals) {
    var keep = nals.filter(function (n) { var t = n[0] & 31; return t !== 9 && t !== 12; });
    var size = 0;
    keep.forEach(function (n) { size += 4 + n.length; });
    var out = new Uint8Array(size), o = 0;
    keep.forEach(function (n) {
      out[o] = n.length >>> 24; out[o + 1] = (n.length >>> 16) & 255; out[o + 2] = (n.length >>> 8) & 255; out[o + 3] = n.length & 255;
      out.set(n, o + 4);
      o += 4 + n.length;
    });
    return out;
  }

  function avcC(sps, pps) {
    var s = sps.nal, high = HIGH_PROFILES[sps.profile] && sps.profile !== 66 && sps.profile !== 77 && sps.profile !== 88;
    var len = 6 + 2 + s.length + 1 + 2 + pps.length + (high ? 4 : 0), out = new Uint8Array(len), o = 0;
    out[o++] = 1; out[o++] = s[1]; out[o++] = s[2]; out[o++] = s[3]; out[o++] = 0xff; out[o++] = 0xe1;
    out[o++] = s.length >> 8; out[o++] = s.length & 255; out.set(s, o); o += s.length;
    out[o++] = 1; out[o++] = pps.length >> 8; out[o++] = pps.length & 255; out.set(pps, o); o += pps.length;
    if (high) {
      out[o++] = 0xfc | sps.chromaFormat;
      out[o++] = 0xf8 | (sps.bitDepthLuma - 8);
      out[o++] = 0xf8 | (sps.bitDepthChroma - 8);
      out[o++] = 0;
    }
    return out;
  }
  function hex2(v) { return (v < 16 ? '0' : '') + v.toString(16); }
  function codecString(sps) { return 'avc1.' + hex2(sps.nal[1]) + hex2(sps.nal[2]) + hex2(sps.nal[3]); }

  /* ---------- transport stream ---------- */

  function readPts(b, i) {
    return ((b[i] >> 1) & 7) * 1073741824 + (b[i + 1] << 22 | (b[i + 2] >> 1) << 15 | b[i + 3] << 7 | b[i + 4] >> 1);
  }
  var WRAP = 8589934592; // 2^33
  function Unwrapper() { this.last = null; this.offset = 0; }
  Unwrapper.prototype.fix = function (v) {
    if (v == null) return v;
    v += this.offset;
    if (this.last !== null && v < this.last - WRAP / 2) { this.offset += WRAP; v += WRAP; }
    this.last = v;
    return v;
  };

  function Demuxer(size, cb) {
    this.size = size; this.skip = size === 192 ? 4 : 0; this.cb = cb;
    this.carry = null; this.pmtPid = -1; this.videoPid = -1; this.audio = {}; this.streams = null;
    this.pes = null; this.unwrap = new Unwrapper();
    this.nals = []; this.auPts = null; this.auDts = null; this.seenVcl = false; this.pendingPts = null;
  }
  Demuxer.prototype.push = function (chunk) {
    var b = chunk, size = this.size;
    if (this.carry) { b = new Uint8Array(this.carry.length + chunk.length); b.set(this.carry); b.set(chunk, this.carry.length); this.carry = null; }
    var i = 0;
    while (i + size <= b.length) {
      var p = i + this.skip;
      if (b[p] !== 0x47) {
        // lost sync: find the next sync byte that repeats a packet later
        var j = p + 1;
        while (j + size < b.length && !(b[j] === 0x47 && b[j + size] === 0x47)) j++;
        i = j - this.skip;
        continue;
      }
      this.packet(b, p);
      i += size;
    }
    if (i < b.length) this.carry = b.slice(i);
  };
  Demuxer.prototype.packet = function (b, p) {
    var pusi = b[p + 1] & 0x40, pid = ((b[p + 1] & 31) << 8) | b[p + 2], afc = (b[p + 3] >> 4) & 3, o = p + 4;
    if (afc === 2 || afc === 0) return;
    if (afc === 3) o += 1 + b[o];
    var end = p + 188;
    if (o >= end) return;
    if (pid === 0) { if (pusi) this.pat(b, o + 1 + b[o], end); return; }
    if (pid === this.pmtPid) { if (pusi) this.pmt(b, o + 1 + b[o], end); return; }
    if (pid === this.videoPid) { this.videoPayload(b.subarray(o, end), pusi); return; }
    if (this.audio[pid] && pusi && !this.audio[pid].seen) {
      var a = b.subarray(o, end);
      if (a[0] === 0 && a[1] === 0 && a[2] === 1 && (a[7] & 0x80)) {
        this.audio[pid].seen = true;
        if (this.cb.onAudioPts) this.cb.onAudioPts(pid, readPts(a, 9));
      }
    }
  };
  Demuxer.prototype.pat = function (b, o, end) {
    var sectionEnd = Math.min(end, o + 3 + (((b[o + 1] & 15) << 8) | b[o + 2]) - 4);
    for (var i = o + 8; i + 4 <= sectionEnd; i += 4) {
      var program = (b[i] << 8) | b[i + 1], pid = ((b[i + 2] & 31) << 8) | b[i + 3];
      if (program !== 0) { this.pmtPid = pid; return; }
    }
  };
  Demuxer.prototype.pmt = function (b, o, end) {
    if (this.streams) return;
    var sectionEnd = Math.min(end, o + 3 + (((b[o + 1] & 15) << 8) | b[o + 2]) - 4);
    var infoLen = ((b[o + 10] & 15) << 8) | b[o + 11], i = o + 12 + infoLen, streams = [];
    while (i + 5 <= sectionEnd) {
      var type = b[i], pid = ((b[i + 1] & 31) << 8) | b[i + 2], esLen = ((b[i + 3] & 15) << 8) | b[i + 4];
      var kind = STREAM_KINDS[type] || 'other';
      if (type === 0x06) {
        // private data: AC-3 / E-AC-3 are flagged with descriptors 0x6a / 0x7a
        for (var d = i + 5; d + 2 <= i + 5 + esLen; d += 2 + b[d + 1]) {
          if (b[d] === 0x6a) kind = 'ac3'; else if (b[d] === 0x7a) kind = 'eac3';
        }
      }
      streams.push({ pid: pid, type: type, kind: kind });
      if (kind === 'h264' && this.videoPid < 0) this.videoPid = pid;
      if (AUDIO[kind]) this.audio[pid] = { kind: kind, seen: false };
      i += 5 + esLen;
    }
    this.streams = streams;
    if (this.cb.onStreams) this.cb.onStreams(streams);
  };
  Demuxer.prototype.videoPayload = function (d, pusi) {
    if (pusi) {
      this.flushPes();
      if (!(d[0] === 0 && d[1] === 0 && d[2] === 1)) return;
      var flags = d[7], hdrLen = d[8], pts = null, dts = null;
      if (flags & 0x80) pts = readPts(d, 9);
      if (flags & 0x40) dts = readPts(d, 14);
      this.pes = { parts: [d.slice(9 + hdrLen)], pts: pts, dts: dts == null ? pts : dts };
    } else if (this.pes) {
      this.pes.parts.push(d.slice());
    }
  };
  Demuxer.prototype.flushPes = function () {
    var pes = this.pes;
    if (!pes) return;
    this.pes = null;
    var n = 0;
    pes.parts.forEach(function (x) { n += x.length; });
    var all = new Uint8Array(n), o = 0;
    pes.parts.forEach(function (x) { all.set(x, o); o += x.length; });
    if (pes.pts != null) {
      // keep DTS on the same unwrapped scale as PTS
      var pts = this.unwrap.fix(pes.pts);
      this.pendingPts = { pts: pts, dts: pts - ((pes.pts - pes.dts + WRAP) % WRAP) };
    }
    var nals = splitNals(all);
    // a start code split across PES packets would leave a fragment; treat the first piece of
    // a PES without a start code as the tail of the previous NAL
    if (all.length >= 3 && !(all[0] === 0 && all[1] === 0 && (all[2] === 1 || (all[2] === 0 && all[3] === 1))) && this.nals.length) {
      var head = all.subarray(0, (nals.length ? indexOfStart(all) : all.length));
      var prev = this.nals[this.nals.length - 1], joined = new Uint8Array(prev.length + head.length);
      joined.set(prev); joined.set(head, prev.length);
      this.nals[this.nals.length - 1] = joined;
    }
    for (var i = 0; i < nals.length; i++) this.nal(nals[i]);
  };
  function indexOfStart(b) {
    for (var i = 0; i + 2 < b.length; i++) if (b[i] === 0 && b[i + 1] === 0 && b[i + 2] === 1) return trimEnd(b, i);
    return b.length;
  }
  function trimEnd(b, i) { while (i > 0 && b[i - 1] === 0) i--; return i; }

  // Groups NAL units into access units (H.264 7.4.1.2.3).
  Demuxer.prototype.nal = function (nal) {
    var t = nal[0] & 31, vcl = t === 1 || t === 5;
    var starts = t === 9 || (this.seenVcl && ((t >= 6 && t <= 8) || (t >= 14 && t <= 18))) || (this.seenVcl && vcl && firstMbInSlice(nal) === 0);
    if (starts && this.nals.length) this.emit();
    if (!this.nals.length && this.pendingPts) { this.auPts = this.pendingPts.pts; this.auDts = this.pendingPts.dts; this.pendingPts = null; }
    this.nals.push(nal);
    if (vcl) this.seenVcl = true;
  };
  Demuxer.prototype.emit = function () {
    if (this.seenVcl && this.cb.onAccessUnit) this.cb.onAccessUnit({ nals: this.nals, pts: this.auPts, dts: this.auDts });
    this.nals = []; this.seenVcl = false; this.auPts = null; this.auDts = null;
  };
  Demuxer.prototype.end = function () {
    this.flushPes();
    if (this.nals.length) this.emit();
  };

  /* ---------- frames from access units ---------- */

  function FramePairer(onSample) {
    this.onSample = onSample; this.sps = null; this.pps = null; this.pending = null;
  }
  FramePairer.prototype.push = function (au) {
    var vcl = null, key = false, recovery = false;
    for (var i = 0; i < au.nals.length; i++) {
      var n = au.nals[i], t = n[0] & 31;
      if (t === 7) { try { this.sps = parseSps(n); } catch (e) {} }
      else if (t === 8 && !this.pps) this.pps = n;
      else if (t === 6 && seiHasRecoveryPoint(n)) recovery = true;
      else if ((t === 1 || t === 5) && !vcl) vcl = n;
    }
    if (!vcl || !this.sps) return; // nothing decodable before the first SPS
    var h;
    try { h = parseSliceHeader(vcl, this.sps); } catch (e) { h = { type: vcl[0] & 31, sliceType: -1, field: 0 }; }
    key = h.type === 5 || (h.sliceType === 2 && recovery);
    var unit = { nals: au.nals, pts: au.pts, dts: au.dts, key: key, field: h.field, bottom: h.bottom, frameNum: h.frameNum };
    if (!h.field) { this.flushPending(); this.out(unit); return; }
    var p = this.pending;
    if (p && p.frameNum === unit.frameNum && p.bottom !== unit.bottom) {
      this.pending = null;
      this.out({ nals: p.nals.concat(unit.nals), pts: p.pts, dts: p.dts, key: p.key });
      return;
    }
    this.flushPending();
    this.pending = unit;
  };
  FramePairer.prototype.flushPending = function () {
    if (this.pending) { var p = this.pending; this.pending = null; this.out(p); }
  };
  FramePairer.prototype.out = function (u) {
    this.onSample({ nals: u.nals, pts: u.pts, dts: u.dts, key: u.key, sps: this.sps, pps: this.pps });
  };
  FramePairer.prototype.end = function () { this.flushPending(); };

  var MTS = {
    packetSize: packetSize, Demuxer: Demuxer, FramePairer: FramePairer,
    parseSps: parseSps, parseSliceHeader: parseSliceHeader, splitNals: splitNals,
    avcC: avcC, toLengthPrefixed: toLengthPrefixed, codecString: codecString, STREAM_KINDS: STREAM_KINDS
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = MTS;
  else root.MTS = MTS;
})(this);
