/* eslint-disable @next/next/no-sync-scripts */
"use client";

import { useEffect, useRef, useState } from "react";

type WsMsg =
  | { type: "transcript"; text: string }
  | { type: "tts_begin"; format: "pcm_s16le"; sample_rate: number }
  | { type: "tts_first_token" }
  | { type: "tts_end" }
  | {
      type: "graph_result";
      pizza_type: string;
      messages: { role: string; content: string }[];
      interrupt?: any;
    }
  | { type: "error"; error: string };

function pcmToWavBlob(pcm: Int16Array, sampleRate: number): Blob {
  const numChannels = 1;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = pcm.byteLength;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);
  new Uint8Array(buffer, 44).set(new Uint8Array(pcm.buffer));
  return new Blob([buffer], { type: "audio/wav" });
}

async function blobToBase64(blob: Blob): Promise<string> {
  const buf = await blob.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function bytesToInt16(bytes: Uint8Array): Int16Array {
  const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return new Int16Array(buf);
}

function bytesToInt16View(bytes: Uint8Array): Int16Array {
  if (bytes.byteLength % 2 === 0 && bytes.byteOffset % 2 === 0) {
    return new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  }
  return bytesToInt16(bytes);
}

// ─── Talking Pizza SVG ─────────────────────────────────────────────────────────
function TalkingPizza({ isTalking, className }: { isTalking: boolean; className?: string }) {
  return (
    <svg viewBox="0 0 400 400" className={className} xmlns="http://www.w3.org/2000/svg">
      {/* Pizza body - triangle slice */}
      <g className={isTalking ? "pizza-talking" : ""} style={{ transformOrigin: "200px 200px" }}>
        {/* Crust arc */}
        <path
          d="M 80 280 Q 200 60 320 280 Z"
          fill="#F5C342"
          stroke="#D4A017"
          strokeWidth="4"
        />
        {/* Crust edge */}
        <path
          d="M 80 280 Q 200 240 320 280"
          fill="#D4A017"
          stroke="#B8860B"
          strokeWidth="3"
        />
        {/* Sauce layer */}
        <path
          d="M 110 265 Q 200 90 290 265 Z"
          fill="#CC3333"
          opacity="0.7"
        />
        {/* Cheese spots */}
        <circle cx="170" cy="200" r="18" fill="#FFE4A0" opacity="0.6" />
        <circle cx="230" cy="190" r="15" fill="#FFE4A0" opacity="0.5" />
        <circle cx="200" cy="150" r="12" fill="#FFE4A0" opacity="0.6" />

        {/* Pepperoni */}
        <circle cx="155" cy="215" r="14" fill="#CC2200" />
        <circle cx="245" cy="210" r="12" fill="#CC2200" />
        <circle cx="200" cy="160" r="11" fill="#CC2200" />
        <circle cx="180" cy="175" r="10" fill="#CC2200" />
        <circle cx="225" cy="170" r="9" fill="#CC2200" />

        {/* Eyes */}
        <g>
          {/* Left eye */}
          <ellipse cx="170" cy="155" rx="16" ry="18" fill="white" />
          <ellipse cx="173" cy="157" rx="8" ry="10" fill="#222" />
          <circle cx="176" cy="153" r="3" fill="white" />
          {/* Right eye */}
          <ellipse cx="230" cy="155" rx="16" ry="18" fill="white" />
          <ellipse cx="233" cy="157" rx="8" ry="10" fill="#222" />
          <circle cx="236" cy="153" r="3" fill="white" />
        </g>

        {/* Mouth */}
        {isTalking ? (
          <g>
            <ellipse cx="200" cy="230" rx="30" ry="18" fill="#8B0000" />
            <ellipse cx="200" cy="225" rx="30" ry="6" fill="none" stroke="#222" strokeWidth="2" />
            <ellipse cx="200" cy="238" rx="12" ry="5" fill="#FF6B6B" />
          </g>
        ) : (
          <g>
            <path d="M 170 225 Q 200 245 230 225" fill="none" stroke="#222" strokeWidth="3" strokeLinecap="round" />
          </g>
        )}

        {/* Red Hat */}
        <g>
          <ellipse cx="200" cy="108" rx="60" ry="10" fill="#CC0000" />
          <rect x="155" y="75" width="90" height="35" rx="6" fill="#EE0000" />
          <rect x="145" y="100" width="110" height="14" rx="4" fill="#CC0000" />
          {/* Hat brim highlight */}
          <rect x="160" y="80" width="40" height="4" rx="2" fill="#FF4444" opacity="0.5" />
        </g>
      </g>
    </svg>
  );
}

// ─── Main Component ─────────────────────────────────────────────────────────────
export default function Home() {
  const [wsUrl, setWsUrl] = useState("ws://127.0.0.1:8765");
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<string>("idle");
  const [transcript, setTranscript] = useState<string>("");
  const [pizzaType, setPizzaType] = useState<string>("");
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [error, setError] = useState<string>("");
  const [textToSend, setTextToSend] = useState<string>("Can I order a pepperoni pizza?");
  const [ttsStreamStatus, setTtsStreamStatus] = useState<string>("idle");
  const [ttsStreamBufferedMs, setTtsStreamBufferedMs] = useState<number>(0);
  const [ttsStreamChunks, setTtsStreamChunks] = useState<number>(0);
  const [ttsStreamBytes, setTtsStreamBytes] = useState<number>(0);
  const [ttsStreamFrames, setTtsStreamFrames] = useState<number>(0);
  const [ttsOutSampleRate, setTtsOutSampleRate] = useState<number>(0);
  const [ttsStreamMinBufferedMs, setTtsStreamMinBufferedMs] = useState<number>(0);
  const [ttsStreamMaxBufferedMs, setTtsStreamMaxBufferedMs] = useState<number>(0);
  const [ttsStreamUnderruns, setTtsStreamUnderruns] = useState<number>(0);
  const [ttsStreamRebuffers, setTtsStreamRebuffers] = useState<number>(0);
  const [ttsTtftMs, setTtsTtftMs] = useState<number>(0);
  const [ttsTtfbMs, setTtsTtfbMs] = useState<number>(0);
  const [ttsRecordEnabled, setTtsRecordEnabled] = useState<boolean>(true);
  const [ttsRecordedUrl, setTtsRecordedUrl] = useState<string>("");
  const [ttsRecordedFilename, setTtsRecordedFilename] = useState<string>("");
  const [ttsRecordedSampleRate, setTtsRecordedSampleRate] = useState<number>(0);
  const [ttsRecordedDurationMs, setTtsRecordedDurationMs] = useState<number>(0);
  const [ttsRecordedBytes, setTtsRecordedBytes] = useState<number>(0);
  const [micDevices, setMicDevices] = useState<MediaDeviceInfo[]>([]);
  const [micDeviceId, setMicDeviceId] = useState<string>("default");
  const [controlsOpen, setControlsOpen] = useState(false);
  const [waiting, setWaiting] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const ttsReceivingBinaryRef = useRef<boolean>(false);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const pcmRef = useRef<Int16Array[]>([]);

  const sampleRate = 16000;

  // ===== Streaming TTS player (AudioWorklet ring buffer) =====
  const ttsCtxRef = useRef<AudioContext | null>(null);
  const ttsSampleRateRef = useRef<number>(24000);
  const ttsStartedRef = useRef<boolean>(false);
  const ttsByteRemainderRef = useRef<Uint8Array>(new Uint8Array(0));
  const ttsPrebufferMs = 2000;
  const ttsLowWaterMs = 20;
  const ttsHighWaterMs = 400;
  const ttsRebufferHoldMs = 150;
  const ttsRebufferCooldownMs = 2000;
  const ttsEmergencyLowMs = 10;

  const ttsWorkletNodeRef = useRef<AudioWorkletNode | null>(null);
  const ttsWorkletModuleUrlRef = useRef<string>("");
  const ttsWorkletBufferedFramesRef = useRef<number>(0);
  const ttsWorkletUnderrunsRef = useRef<number>(0);
  const ttsWorkletRebuffersRef = useRef<number>(0);
  const ttsWorkletPlayingRef = useRef<boolean>(false);
  const ttsStatsLatchedRef = useRef<boolean>(false);
  const ttsReqStartedAtMsRef = useRef<number>(0);
  const ttsFirstTokenAtMsRef = useRef<number>(0);
  const ttsFirstAudioAtMsRef = useRef<number>(0);

  const ttsSabAudioRef = useRef<SharedArrayBuffer | null>(null);
  const ttsSabCtrlRef = useRef<SharedArrayBuffer | null>(null);
  const ttsSabAudioI16Ref = useRef<Int16Array | null>(null);
  const ttsSabCtrlI32Ref = useRef<Int32Array | null>(null);
  const ttsSabSamplesRef = useRef<number>(0);

  const ttsBufferedMsRef = useRef<number>(0);
  const ttsMinBufferedMsRef = useRef<number>(Number.POSITIVE_INFINITY);
  const ttsMaxBufferedMsRef = useRef<number>(0);
  const ttsStreamStatusRef = useRef<string>("idle");

  const ttsRecordedChunksRef = useRef<Int16Array[]>([]);
  const ttsRecordedSamplesRef = useRef<number>(0);
  const ttsRecordedBuffersRef = useRef<ArrayBuffer[]>([]);
  const ttsRecordedBytesLenRef = useRef<number>(0);
  const ttsStreamBytesRef = useRef<number>(0);
  const ttsStreamChunksRef = useRef<number>(0);
  const ttsStreamFramesRef = useRef<number>(0);
  const ttsRxStartedAtMsRef = useRef<number>(0);
  const ttsRxLastAtMsRef = useRef<number>(0);
  const ttsRxInSamplesRef = useRef<number>(0);
  const [ttsGenRealtimeX, setTtsGenRealtimeX] = useState<number>(0);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const _ttsOutRate = () => ttsCtxRef.current?.sampleRate ?? ttsSampleRateRef.current;

  const isSpeaking = ttsStreamStatus === "buffering" || ttsStreamStatus === "draining";

  // Auto-scroll conversation
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, waiting]);

  const stopTtsStream = (opts?: { resetStats?: boolean }) => {
    const resetStats = opts?.resetStats ?? true;
    ttsStartedRef.current = false;
    ttsByteRemainderRef.current = new Uint8Array(0);
    ttsBufferedMsRef.current = 0;
    ttsSabAudioRef.current = null;
    ttsSabCtrlRef.current = null;
    ttsSabAudioI16Ref.current = null;
    ttsSabCtrlI32Ref.current = null;
    ttsSabSamplesRef.current = 0;
    ttsWorkletBufferedFramesRef.current = 0;
    ttsWorkletPlayingRef.current = false;
    try {
      if (resetStats) {
        ttsWorkletNodeRef.current?.port.postMessage({ type: "reset" });
      } else {
        ttsWorkletNodeRef.current?.port.postMessage({ type: "stop" });
      }
    } catch {}
    try {
      ttsWorkletNodeRef.current?.disconnect();
    } catch {}
    ttsWorkletNodeRef.current = null;
    setTtsStreamBufferedMs(0);
    if (resetStats) {
      ttsMinBufferedMsRef.current = Number.POSITIVE_INFINITY;
      ttsMaxBufferedMsRef.current = 0;
      setTtsStreamMinBufferedMs(0);
      setTtsStreamMaxBufferedMs(0);
      ttsWorkletUnderrunsRef.current = 0;
      ttsWorkletRebuffersRef.current = 0;
      setTtsStreamUnderruns(0);
      setTtsStreamRebuffers(0);
    }
    setTtsStreamStatus("idle");
    setWaiting(false);
  };

  const clearTtsRecording = () => {
    setTtsRecordedUrl("");
    setTtsRecordedFilename("");
    setTtsRecordedSampleRate(0);
    setTtsRecordedDurationMs(0);
    setTtsRecordedBytes(0);
    ttsRecordedChunksRef.current = [];
    ttsRecordedSamplesRef.current = 0;
    ttsRecordedBuffersRef.current = [];
    ttsRecordedBytesLenRef.current = 0;
  };

  const finalizeTtsRecording = () => {
    if (!ttsRecordEnabled) return;
    const sr = ttsRecordedSampleRate || ttsSampleRateRef.current;
    if (!sr) return;
    const bufs = ttsRecordedBuffersRef.current;
    const totalBytes = ttsRecordedBytesLenRef.current;
    let wav: Blob | null = null;
    let durationMs = 0;
    if (bufs.length && totalBytes) {
      const joinedBytes = new Uint8Array(totalBytes);
      let offB = 0;
      for (const b of bufs) {
        joinedBytes.set(new Uint8Array(b), offB);
        offB += b.byteLength;
      }
      const evenLen = joinedBytes.length - (joinedBytes.length % 2);
      const i16 = bytesToInt16View(joinedBytes.subarray(0, evenLen));
      wav = pcmToWavBlob(i16, sr);
      durationMs = (i16.length / sr) * 1000;
    } else {
      const chunks = ttsRecordedChunksRef.current;
      const total = ttsRecordedSamplesRef.current;
      if (!chunks.length || !total) return;
      const joined = new Int16Array(total);
      let off = 0;
      for (const c of chunks) {
        joined.set(c, off);
        off += c.length;
      }
      wav = pcmToWavBlob(joined, sr);
      durationMs = (joined.length / sr) * 1000;
    }
    if (!wav) return;
    const url = URL.createObjectURL(wav);
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    setTtsRecordedUrl(url);
    setTtsRecordedFilename(`tts-stream-${ts}.wav`);
    setTtsRecordedBytes(wav.size);
    setTtsRecordedDurationMs(durationMs);
  };

  const ensureTtsWorklet = async (inRate: number) => {
    const ctx = await ensureTtsContext(inRate);
    if (!("audioWorklet" in ctx)) {
      throw new Error("AudioWorklet not supported in this browser.");
    }
    if (ttsWorkletNodeRef.current) return ttsWorkletNodeRef.current;

    if (!ttsWorkletModuleUrlRef.current) {
      const moduleCode = `
class TtsPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.inRate = 24000;
    this.outRate = sampleRate;
    this.sharedCtrl = null;
    this.sharedAudio = null;
    this.ctrl = null;
    this.audio = null;
    this.audioSamples = 0;
    this.underruns = 0;
    this.rebuffers = 0;
    this.playing = false;
    this.startedOnce = false;
    this.enabled = true;
    this.eos = false;
    this.holdFrames = 0;
    this.holdUntil = 0;
    this.cooldownFrames = 0;
    this.cooldownUntil = 0;
    this.emergencyLowFrames = 0;
    this.outCursor = 0;
    this.startFrames = Math.floor(this.outRate * 0.9);
    this.lowFrames = Math.floor(this.outRate * 0.25);
    this.highFrames = Math.floor(this.outRate * 0.75);
    this.rem = new Float32Array(0);
    this.pos = 0;
    this._tick = 0;
    this.port.onmessage = (e) => {
      const msg = e.data || {};
      if (msg.type === "config" && typeof msg.inRate === "number") {
        this.inRate = msg.inRate;
        this.enabled = true;
        this.eos = false;
        if (typeof msg.startFrames === "number") this.startFrames = Math.max(0, msg.startFrames|0);
        if (typeof msg.lowFrames === "number") this.lowFrames = Math.max(0, msg.lowFrames|0);
        if (typeof msg.highFrames === "number") this.highFrames = Math.max(0, msg.highFrames|0);
        if (typeof msg.holdFrames === "number") this.holdFrames = Math.max(0, msg.holdFrames|0);
        if (typeof msg.cooldownFrames === "number") this.cooldownFrames = Math.max(0, msg.cooldownFrames|0);
        if (typeof msg.emergencyLowFrames === "number") this.emergencyLowFrames = Math.max(0, msg.emergencyLowFrames|0);
      } else if (msg.type === "init_shared" && msg.ctrl && msg.audio && typeof msg.audioSamples === "number") {
        this.sharedCtrl = msg.ctrl;
        this.sharedAudio = msg.audio;
        this.ctrl = new Int32Array(this.sharedCtrl);
        this.audio = new Int16Array(this.sharedAudio);
        this.audioSamples = msg.audioSamples|0;
      } else if (msg.type === "eos") {
        this.eos = true;
      } else if (msg.type === "stop") {
        this.enabled = false;
        this.playing = false;
      } else if (msg.type === "reset") {
        this.underruns = 0;
        this.rebuffers = 0; this.playing = false; this.enabled = true; this.eos = false;
        this.holdUntil = 0;
        this.cooldownUntil = 0;
        this.outCursor = 0;
        this.rem = new Float32Array(0); this.pos = 0;
      }
    };
  }

  _availableInputSamples(w, r, size) {
    return w >= r ? (w - r) : (size - (r - w));
  }

  _readInputSampleAt(pos) {
    return this.audio[pos] / 32768;
  }

  process(inputs, outputs) {
    const out = outputs[0][0];
    const frames = out.length;
    this.outCursor += frames;
    if (!this.ctrl || !this.audio || !this.audioSamples) {
      out.fill(0);
      return true;
    }
    if (!this.enabled) {
      out.fill(0);
      this._tick++;
      if ((this._tick % 20) === 0) {
        this.port.postMessage({ type: "stats", bufferedFrames: 0, underruns: this.underruns, rebuffers: this.rebuffers, playing: this.playing });
      }
      return true;
    }
    const w0 = Atomics.load(this.ctrl, 0);
    const r0 = Atomics.load(this.ctrl, 1);
    const avail0 = this._availableInputSamples(w0, r0, this.audioSamples);
    if (this.eos && avail0 <= 0) {
      this.enabled = false;
      this.playing = false;
      out.fill(0);
      this._tick++;
      if ((this._tick % 20) === 0) {
        this.port.postMessage({ type: "stats", bufferedFrames: 0, underruns: this.underruns, rebuffers: this.rebuffers, playing: this.playing });
      }
      return true;
    }
    if (!this.playing) {
      const bufferedFrames = Math.floor((avail0 * this.outRate) / this.inRate);
      const holdActive = this.startedOnce && this.outCursor < this.holdUntil;
      const startThreshold = this.startedOnce && this.highFrames > 0 ? this.highFrames : this.startFrames;
      if (bufferedFrames >= startThreshold || (startThreshold === 0 && bufferedFrames > 0)) {
        if (!holdActive) {
          this.playing = true;
          this.startedOnce = true;
          if (this.cooldownFrames > 0) this.cooldownUntil = this.outCursor + this.cooldownFrames;
        }
      } else {
        out.fill(0);
        this._tick++;
        if ((this._tick % 20) === 0) {
          this.port.postMessage({ type: "stats", bufferedFrames, underruns: this.underruns, rebuffers: this.rebuffers, playing: this.playing });
        }
        return true;
      }
      if (!this.playing) {
        out.fill(0);
        return true;
      }
    }
    const bufferedFramesNow = Math.floor((avail0 * this.outRate) / this.inRate);
    if (this.lowFrames > 0 && bufferedFramesNow < this.lowFrames) {
      const inCooldown = this.cooldownFrames > 0 && this.outCursor < this.cooldownUntil;
      const emergency = this.emergencyLowFrames > 0 && bufferedFramesNow <= this.emergencyLowFrames;
      if (inCooldown && !emergency) {
      } else {
      this.playing = false;
      this.rebuffers++;
      if (this.holdFrames > 0) this.holdUntil = this.outCursor + this.holdFrames;
      out.fill(0);
      this._tick++;
      if ((this._tick % 20) === 0) {
        this.port.postMessage({ type: "stats", bufferedFrames: bufferedFramesNow, underruns: this.underruns, rebuffers: this.rebuffers, playing: this.playing });
      }
      return true;
      }
    }
    if (!this.playing && this.highFrames > 0 && bufferedFramesNow >= this.highFrames) {
      this.playing = true;
    }
    let w = w0;
    let r = r0;
    let avail = avail0;
    const step = this.inRate / this.outRate;
    let frac = this.pos;
    if (avail < 2) {
      out.fill(0);
      this.underruns++;
      this._tick++;
      if ((this._tick % 20) === 0) {
        const bf = Math.floor((avail * this.outRate) / this.inRate);
        this.port.postMessage({ type: "stats", bufferedFrames: bf, underruns: this.underruns, rebuffers: this.rebuffers, playing: this.playing });
      }
      return true;
    }
    let s0 = this._readInputSampleAt(r);
    let s1 = this._readInputSampleAt((r + 1) % this.audioSamples);
    for (let i = 0; i < frames; i++) {
      out[i] = s0 + (s1 - s0) * frac;
      frac += step;
      while (frac >= 1.0) {
        frac -= 1.0;
        r = (r + 1) % this.audioSamples;
        avail -= 1;
        if (avail < 1) {
          out.fill(0, i + 1);
          this.underruns++;
          break;
        }
        s0 = s1;
        s1 = this._readInputSampleAt((r + 1) % this.audioSamples);
      }
      if (avail < 1) break;
    }
    this.pos = frac;
    Atomics.store(this.ctrl, 1, r);
    this._tick++;
    if ((this._tick % 20) === 0) {
      const w1 = Atomics.load(this.ctrl, 0);
      const r1 = Atomics.load(this.ctrl, 1);
      const avail1 = this._availableInputSamples(w1, r1, this.audioSamples);
      const bufferedFrames = Math.floor((avail1 * this.outRate) / this.inRate);
      this.port.postMessage({ type: "stats", bufferedFrames, underruns: this.underruns, rebuffers: this.rebuffers, playing: this.playing });
    }
    return true;
  }
}
registerProcessor("tts-player", TtsPlayerProcessor);
`;
      const blob = new Blob([moduleCode], { type: "text/javascript" });
      ttsWorkletModuleUrlRef.current = URL.createObjectURL(blob);
    }

    await ctx.audioWorklet.addModule(ttsWorkletModuleUrlRef.current);
    const node = new AudioWorkletNode(ctx, "tts-player", { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1] });
    node.port.onmessage = (e) => {
      const msg = e.data || {};
      if (msg.type === "stats") {
        if (typeof msg.bufferedFrames === "number") ttsWorkletBufferedFramesRef.current = msg.bufferedFrames;
        if (typeof msg.underruns === "number") ttsWorkletUnderrunsRef.current = msg.underruns;
        if (typeof msg.rebuffers === "number") ttsWorkletRebuffersRef.current = msg.rebuffers;
        if (typeof msg.playing === "boolean") ttsWorkletPlayingRef.current = msg.playing;
      }
    };
    node.connect(ctx.destination);
    const startFrames = Math.max(0, Math.floor((ctx.sampleRate * ttsPrebufferMs) / 1000));
    const lowFrames = Math.max(0, Math.floor((ctx.sampleRate * ttsLowWaterMs) / 1000));
    const highFrames = Math.max(lowFrames, Math.floor((ctx.sampleRate * ttsHighWaterMs) / 1000));
    const holdFrames = Math.max(0, Math.floor((ctx.sampleRate * ttsRebufferHoldMs) / 1000));
    const cooldownFrames = Math.max(0, Math.floor((ctx.sampleRate * ttsRebufferCooldownMs) / 1000));
    const emergencyLowFrames = Math.max(0, Math.floor((ctx.sampleRate * ttsEmergencyLowMs) / 1000));
    node.port.postMessage({
      type: "config",
      inRate,
      startFrames,
      lowFrames,
      highFrames,
      holdFrames,
      cooldownFrames,
      emergencyLowFrames,
    });
    ttsWorkletNodeRef.current = node;
    return node;
  };

  const ensureSharedRing = (inRate: number) => {
    const seconds = 12;
    const samples = Math.max(1, Math.floor(inRate * seconds));
    if (ttsSabAudioRef.current && ttsSabSamplesRef.current === samples) return;
    if (typeof SharedArrayBuffer === "undefined" || !(globalThis as any).crossOriginIsolated) {
      throw new Error(
        "SharedArrayBuffer unavailable. Ensure COOP/COEP headers are set (crossOriginIsolated=true)."
      );
    }
    ttsSabSamplesRef.current = samples;
    ttsSabAudioRef.current = new SharedArrayBuffer(samples * 2);
    ttsSabCtrlRef.current = new SharedArrayBuffer(8);
    ttsSabAudioI16Ref.current = new Int16Array(ttsSabAudioRef.current);
    ttsSabCtrlI32Ref.current = new Int32Array(ttsSabCtrlRef.current);
    Atomics.store(ttsSabCtrlI32Ref.current, 0, 0);
    Atomics.store(ttsSabCtrlI32Ref.current, 1, 0);
  };

  const writeToSharedRing = (pcmBuf: ArrayBuffer) => {
    const ring = ttsSabAudioI16Ref.current;
    const ctrl = ttsSabCtrlI32Ref.current;
    if (!ring || !ctrl) return;
    const src = new Int16Array(pcmBuf);
    const size = ring.length;
    let w = Atomics.load(ctrl, 0);
    let r = Atomics.load(ctrl, 1);
    const avail = w >= r ? (w - r) : (size - (r - w));
    let free = size - avail - 1;
    if (src.length > free) {
      const drop = src.length - free;
      r = (r + drop) % size;
      Atomics.store(ctrl, 1, r);
      free += drop;
    }
    const toWrite = Math.min(src.length, free);
    const first = Math.min(toWrite, size - w);
    ring.set(src.subarray(0, first), w);
    const remain = toWrite - first;
    if (remain > 0) ring.set(src.subarray(first, first + remain), 0);
    w = (w + toWrite) % size;
    Atomics.store(ctrl, 0, w);
  };

  useEffect(() => {
    ttsStreamStatusRef.current = ttsStreamStatus;
  }, [ttsStreamStatus]);

  const prevTtsRecordedUrlRef = useRef<string>("");
  useEffect(() => {
    const prev = prevTtsRecordedUrlRef.current;
    if (prev && prev !== ttsRecordedUrl) URL.revokeObjectURL(prev);
    prevTtsRecordedUrlRef.current = ttsRecordedUrl;
    return () => {
      if (ttsRecordedUrl) URL.revokeObjectURL(ttsRecordedUrl);
    };
  }, [ttsRecordedUrl]);

  const ensureTtsContext = async (_sr: number) => {
    if (ttsCtxRef.current) return ttsCtxRef.current;
    ttsCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    setTtsOutSampleRate(ttsCtxRef.current.sampleRate);
    return ttsCtxRef.current!;
  };

  const primeTtsAudio = async () => {
    const ctx = await ensureTtsContext(ttsSampleRateRef.current);
    if (ctx.state !== "running") {
      await ctx.resume();
    }
  };

  useEffect(() => {
    const id = window.setInterval(() => {
      const ctx = ttsCtxRef.current;
      const outRate = ctx?.sampleRate ?? _ttsOutRate();
      const bufferedFrames = ttsWorkletBufferedFramesRef.current || 0;
      const ms = outRate ? (bufferedFrames / outRate) * 1000 : 0;
      ttsBufferedMsRef.current = ms;
      setTtsStreamBufferedMs(ms);
      setTtsStreamBytes(ttsStreamBytesRef.current);
      setTtsStreamChunks(ttsStreamChunksRef.current);
      setTtsStreamFrames(ttsStreamFramesRef.current);
      if (ttsStreamStatusRef.current !== "idle" || ttsWorkletNodeRef.current) {
        if (!ttsStatsLatchedRef.current) {
          setTtsStreamUnderruns(ttsWorkletUnderrunsRef.current || 0);
          setTtsStreamRebuffers(ttsWorkletRebuffersRef.current || 0);
        }
      }
      if (
        ttsStreamStatusRef.current !== "idle" &&
        ttsRxStartedAtMsRef.current > 0 &&
        ttsRxLastAtMsRef.current >= ttsRxStartedAtMsRef.current &&
        ttsSampleRateRef.current > 0
      ) {
        const elapsedS = Math.max(
          0.001,
          (ttsRxLastAtMsRef.current - ttsRxStartedAtMsRef.current) / 1000
        );
        const audioS = ttsRxInSamplesRef.current / ttsSampleRateRef.current;
        setTtsGenRealtimeX(audioS / elapsedS);
      }
      if (
        ttsStreamStatusRef.current !== "idle" &&
        ttsWorkletPlayingRef.current &&
        bufferedFrames > 0
      ) {
        ttsMinBufferedMsRef.current = Math.min(ttsMinBufferedMsRef.current, ms);
        ttsMaxBufferedMsRef.current = Math.max(ttsMaxBufferedMsRef.current, ms);
        setTtsStreamMinBufferedMs(
          Number.isFinite(ttsMinBufferedMsRef.current) ? ttsMinBufferedMsRef.current : 0
        );
        setTtsStreamMaxBufferedMs(ttsMaxBufferedMsRef.current);
      }
    }, 200);
    return () => window.clearInterval(id);
  }, []);

  const connect = () => {
    setError("");
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      wsRef.current = ws;
      setConnected(true);
      setStatus("connected");
    };
    ws.onclose = () => {
      setConnected(false);
      setStatus("disconnected");
    };
    ws.onerror = () => setError("WebSocket error");
    ws.onmessage = (evt) => {
      try {
        if (typeof evt.data !== "string") {
          if (!ttsReceivingBinaryRef.current) return;
          const handleBuffer = (buf: ArrayBuffer) => {
            const len = buf.byteLength;
            if (!len) return;
            ttsStreamBytesRef.current += len;
            ttsStreamChunksRef.current += 1;
            ttsRxLastAtMsRef.current = Date.now();
            if (ttsFirstAudioAtMsRef.current === 0) {
              ttsFirstAudioAtMsRef.current = ttsRxLastAtMsRef.current;
              if (ttsReqStartedAtMsRef.current > 0) {
                setTtsTtfbMs(ttsFirstAudioAtMsRef.current - ttsReqStartedAtMsRef.current);
              }
            }
            const inRate = ttsSampleRateRef.current || 24000;
            const outRate = ttsCtxRef.current?.sampleRate ?? 48000;
            const inSamples = Math.floor(len / 2);
            ttsRxInSamplesRef.current += inSamples;
            ttsStreamFramesRef.current += Math.round((inSamples * outRate) / inRate);
            if (ttsRecordEnabled) {
              const copy = buf.slice(0);
              ttsRecordedBuffersRef.current.push(copy);
              ttsRecordedBytesLenRef.current += copy.byteLength;
            }
            writeToSharedRing(buf);
          };
          if (evt.data instanceof ArrayBuffer) {
            handleBuffer(evt.data);
            return;
          }
          if (evt.data instanceof Blob) {
            void evt.data.arrayBuffer().then((buf) => handleBuffer(buf));
            return;
          }
          return;
        }

        const msg = JSON.parse(evt.data) as WsMsg;
        if (msg.type === "transcript") setTranscript(msg.text);
        if (msg.type === "tts_begin") {
          stopTtsStream({ resetStats: true });
          ttsStatsLatchedRef.current = false;
          ttsSampleRateRef.current = msg.sample_rate;
          ttsReceivingBinaryRef.current = true;
          ttsFirstAudioAtMsRef.current = 0;
          ttsFirstTokenAtMsRef.current = 0;
          setTtsTtftMs(0);
          setTtsTtfbMs(0);
          setTtsStreamStatus("buffering");
          ttsStreamChunksRef.current = 0;
          ttsStreamBytesRef.current = 0;
          ttsStreamFramesRef.current = 0;
          ttsRxStartedAtMsRef.current = Date.now();
          ttsRxLastAtMsRef.current = ttsRxStartedAtMsRef.current;
          ttsRxInSamplesRef.current = 0;
          setTtsStreamChunks(0);
          setTtsStreamBytes(0);
          setTtsStreamFrames(0);
          ttsMinBufferedMsRef.current = Number.POSITIVE_INFINITY;
          ttsMaxBufferedMsRef.current = 0;
          setTtsStreamMinBufferedMs(0);
          setTtsStreamMaxBufferedMs(0);
          ttsWorkletUnderrunsRef.current = 0;
          ttsWorkletRebuffersRef.current = 0;
          setTtsStreamUnderruns(0);
          setTtsStreamRebuffers(0);
          if (ttsRecordEnabled) {
            clearTtsRecording();
            setTtsRecordedSampleRate(msg.sample_rate);
          }
          void ensureTtsContext(ttsSampleRateRef.current);
          try {
            ensureSharedRing(msg.sample_rate);
            void ensureTtsWorklet(msg.sample_rate)
              .then((node) => {
                node.port.postMessage({
                  type: "init_shared",
                  ctrl: ttsSabCtrlRef.current,
                  audio: ttsSabAudioRef.current,
                  audioSamples: ttsSabSamplesRef.current,
                });
              })
              .catch((e) => setError(e?.message || "Failed to init AudioWorklet"));
          } catch (e: any) {
            setError(
              e?.message ||
                "SharedArrayBuffer unavailable. Ensure COOP/COEP headers are set (crossOriginIsolated=true)."
            );
          }
        }
        if (msg.type === "tts_first_token") {
          if (ttsFirstTokenAtMsRef.current === 0) {
            ttsFirstTokenAtMsRef.current = Date.now();
            if (ttsReqStartedAtMsRef.current > 0) {
              setTtsTtftMs(ttsFirstTokenAtMsRef.current - ttsReqStartedAtMsRef.current);
            }
          }
        }
        if (msg.type === "tts_end") {
          setTtsStreamStatus("draining");
          ttsReceivingBinaryRef.current = false;
          ttsByteRemainderRef.current = new Uint8Array(0);
          try {
            ttsWorkletNodeRef.current?.port.postMessage({ type: "eos" });
          } catch {}
          ttsStatsLatchedRef.current = true;
          setTtsStreamUnderruns(ttsWorkletUnderrunsRef.current || 0);
          setTtsStreamRebuffers(ttsWorkletRebuffersRef.current || 0);
          if (ttsReqStartedAtMsRef.current > 0 && ttsFirstTokenAtMsRef.current > 0) {
            setTtsTtftMs(ttsFirstTokenAtMsRef.current - ttsReqStartedAtMsRef.current);
          }
          if (ttsReqStartedAtMsRef.current > 0 && ttsFirstAudioAtMsRef.current > 0) {
            setTtsTtfbMs(ttsFirstAudioAtMsRef.current - ttsReqStartedAtMsRef.current);
          }
          finalizeTtsRecording();
          const check = setInterval(() => {
            const ctx = ttsCtxRef.current;
            const outRate = ctx?.sampleRate ?? _ttsOutRate();
            const bufferedFrames = ttsWorkletBufferedFramesRef.current || 0;
            const ms = outRate ? (bufferedFrames / outRate) * 1000 : 0;
            if (ttsStartedRef.current && ms <= 5) {
              stopTtsStream({ resetStats: false });
              clearInterval(check);
            }
          }, 200);
        }
        if (msg.type === "graph_result") {
          setPizzaType(msg.pizza_type);
          setMessages(msg.messages);
        }
        if (msg.type === "error") setError(msg.error);
      } catch (e) {
        console.error("WS message handling failed:", e);
      }
    };
  };

  const disconnect = () => {
    wsRef.current?.close();
    wsRef.current = null;
    setConnected(false);
  };

  const startRecording = async () => {
    setError("");
    setTranscript("");
    pcmRef.current = [];
    if (!connected || !wsRef.current) {
      setError("Connect to WS server first.");
      return;
    }
    setStatus("requesting mic\u2026");
    try {
      const constraints: MediaStreamConstraints =
        micDeviceId && micDeviceId !== "default"
          ? { audio: { deviceId: { exact: micDeviceId } } }
          : { audio: true };
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (err: any) {
        if (micDeviceId && micDeviceId !== "default") {
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          setMicDeviceId("default");
        } else {
          throw err;
        }
      }
      mediaStreamRef.current = stream;
    } catch (e: any) {
      setStatus("mic permission denied");
      setError(e?.message || "Microphone permission denied/unavailable in this browser.");
      return;
    }
    const stream = mediaStreamRef.current!;
    const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
    audioCtxRef.current = audioCtx;
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;
    source.connect(processor);
    processor.connect(audioCtx.destination);
    setStatus("recording");
    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      const inRate = e.inputBuffer.sampleRate;
      const ratio = inRate / sampleRate;
      const outLen = Math.floor(input.length / ratio);
      const out = new Int16Array(outLen);
      for (let i = 0; i < outLen; i++) {
        const idx = Math.floor(i * ratio);
        const s = Math.max(-1, Math.min(1, input[idx]));
        out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      pcmRef.current.push(out);
    };
  };

  const stopAndSend = async () => {
    setStatus("stopping");
    processorRef.current?.disconnect();
    processorRef.current = null;
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    mediaStreamRef.current = null;
    const chunks = pcmRef.current;
    const total = chunks.reduce((acc, c) => acc + c.length, 0);
    const joined = new Int16Array(total);
    let offset = 0;
    for (const c of chunks) {
      joined.set(c, offset);
      offset += c.length;
    }
    const wav = pcmToWavBlob(joined, sampleRate);
    const b64 = await blobToBase64(wav);
    setStatus("sending");
    setWaiting(true);
    wsRef.current?.send(JSON.stringify({ type: "audio_wav_b64", audio_b64: b64 }));
    setStatus("sent (awaiting response)");
  };

  useEffect(() => {
    return () => {
      disconnect();
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
      audioCtxRef.current?.close();
      if (ttsWorkletModuleUrlRef.current) {
        try {
          URL.revokeObjectURL(ttsWorkletModuleUrlRef.current);
        } catch {}
        ttsWorkletModuleUrlRef.current = "";
      }
      ttsCtxRef.current?.close();
    };
  }, []);

  const sendText = () => {
    setError("");
    if (!connected || !wsRef.current) {
      setError("Connect to WS server first.");
      return;
    }
    ttsReqStartedAtMsRef.current = Date.now();
    ttsFirstTokenAtMsRef.current = 0;
    ttsFirstAudioAtMsRef.current = 0;
    setTtsTtftMs(0);
    setTtsTtfbMs(0);
    void primeTtsAudio();
    setWaiting(true);
    wsRef.current.send(JSON.stringify({ type: "text", text: textToSend }));
  };

  const speakStream = () => {
    setError("");
    if (!connected || !wsRef.current) {
      setError("Connect to WS server first.");
      return;
    }
    ttsReqStartedAtMsRef.current = Date.now();
    ttsFirstTokenAtMsRef.current = 0;
    ttsFirstAudioAtMsRef.current = 0;
    setTtsTtftMs(0);
    setTtsTtfbMs(0);
    void primeTtsAudio();
    wsRef.current.send(JSON.stringify({ type: "tts_text", text: textToSend }));
  };

  const sendWavFile = async (file: File) => {
    setError("");
    if (!connected || !wsRef.current) {
      setError("Connect to WS server first.");
      return;
    }
    ttsReqStartedAtMsRef.current = Date.now();
    ttsFirstTokenAtMsRef.current = 0;
    ttsFirstAudioAtMsRef.current = 0;
    setTtsTtftMs(0);
    setTtsTtfbMs(0);
    const b64 = await blobToBase64(file);
    wsRef.current.send(JSON.stringify({ type: "audio_wav_b64", audio_b64: b64 }));
  };

  useEffect(() => {
    const loadDevices = async () => {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        setMicDevices(devices.filter((d) => d.kind === "audioinput"));
      } catch {}
    };
    void loadDevices();
  }, []);

  // Helper: get the last agent message for the transcript bubble
  const lastAgentMessage = [...messages].reverse().find(
    (m) => m.role !== "human" && m.role !== "interrupt" && !m.content.startsWith("Routing to")
  );

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-rh-gray-95 text-rh-gray-10">
      {/* ─── Nav Bar ─────────────────────────────────────────────────── */}
      <nav className="flex-none h-14 flex items-center px-6 border-b border-rh-gray-80 bg-rh-gray-95">
        <div className="flex items-center gap-3">
          {/* Red Hat icon */}
          <div className="flex items-center gap-1">
            <div className="w-5 h-1 bg-rh-red rounded-sm" />
            <div className="w-2 h-2 bg-rh-red rounded-sm" />
          </div>
          <h1 className="text-lg font-bold" style={{ fontFamily: "'Red Hat Display', sans-serif" }}>
            Red Hat Pizza Shop
          </h1>
        </div>
        <div className="ml-auto flex items-center gap-3">
          {/* Connection indicator */}
          <div className="flex items-center gap-2 text-xs">
            <div className={`w-2 h-2 rounded-full ${connected ? "bg-green-500" : "bg-rh-gray-50"}`} />
            <span className="text-rh-gray-40">{connected ? "Connected" : "Disconnected"}</span>
          </div>
          {/* Controls toggle */}
          <button
            onClick={() => setControlsOpen(!controlsOpen)}
            className="text-xs px-3 py-1.5 rounded border border-rh-gray-70 text-rh-gray-40 hover:text-white hover:border-rh-gray-50 transition-colors"
          >
            Controls {controlsOpen ? "\u25B2" : "\u25BC"}
          </button>
        </div>
      </nav>

      {/* ─── Controls Panel (collapsible) ────────────────────────────── */}
      {controlsOpen && (
        <div className="flex-none border-b border-rh-gray-80 bg-rh-gray-90 px-6 py-4 animate-fade-in-up">
          <div className="max-w-5xl mx-auto grid gap-4 md:grid-cols-3">
            {/* Connection */}
            <div className="space-y-2">
              <label className="text-xs text-rh-gray-40 uppercase tracking-wider font-semibold">WebSocket URL</label>
              <input
                className="w-full rounded bg-rh-gray-95 border border-rh-gray-70 px-3 py-2 text-sm text-white placeholder-rh-gray-50 focus:border-rh-red focus:ring-1 focus:ring-rh-red focus:outline-none"
                value={wsUrl}
                onChange={(e) => setWsUrl(e.target.value)}
              />
              <div className="flex gap-2">
                <button
                  className="rounded bg-rh-red text-white px-3 py-1.5 text-xs font-semibold hover:bg-rh-red-dark disabled:opacity-40 transition-colors"
                  onClick={connect}
                  disabled={connected}
                >
                  Connect
                </button>
                <button
                  className="rounded border border-rh-gray-70 text-rh-gray-40 px-3 py-1.5 text-xs hover:text-white disabled:opacity-40 transition-colors"
                  onClick={disconnect}
                  disabled={!connected}
                >
                  Disconnect
                </button>
              </div>
            </div>

            {/* Microphone */}
            <div className="space-y-2">
              <label className="text-xs text-rh-gray-40 uppercase tracking-wider font-semibold">Microphone</label>
              <select
                className="w-full rounded bg-rh-gray-95 border border-rh-gray-70 px-3 py-2 text-sm text-white focus:border-rh-red focus:outline-none"
                value={micDeviceId}
                onChange={(e) => setMicDeviceId(e.target.value)}
              >
                <option value="default">Default</option>
                {micDevices.map((d) => (
                  <option key={d.deviceId} value={d.deviceId}>
                    {d.label || `Mic (${d.deviceId.slice(0, 8)}\u2026)`}
                  </option>
                ))}
              </select>
            </div>

            {/* Quick Test */}
            <div className="space-y-2">
              <label className="text-xs text-rh-gray-40 uppercase tracking-wider font-semibold">Quick Test</label>
              <textarea
                className="w-full min-h-[56px] rounded bg-rh-gray-95 border border-rh-gray-70 px-3 py-2 text-sm text-white placeholder-rh-gray-50 focus:border-rh-red focus:outline-none resize-none"
                value={textToSend}
                onChange={(e) => setTextToSend(e.target.value)}
                rows={2}
              />
              <div className="flex gap-2">
                <button
                  className="rounded bg-rh-red text-white px-3 py-1.5 text-xs font-semibold hover:bg-rh-red-dark disabled:opacity-40 transition-colors"
                  onClick={sendText}
                  disabled={!connected}
                >
                  Send Text
                </button>
                <button
                  className="rounded border border-rh-gray-70 text-rh-gray-40 px-3 py-1.5 text-xs hover:text-white disabled:opacity-40 transition-colors"
                  onClick={speakStream}
                  disabled={!connected}
                >
                  TTS Only
                </button>
                <label className="rounded border border-rh-gray-70 text-rh-gray-40 px-3 py-1.5 text-xs hover:text-white cursor-pointer transition-colors">
                  Upload WAV
                  <input
                    type="file"
                    accept="audio/wav"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) void sendWavFile(f);
                    }}
                    disabled={!connected}
                  />
                </label>
              </div>
            </div>
          </div>

          {/* Playback stats row */}
          <div className="max-w-5xl mx-auto mt-3 pt-3 border-t border-rh-gray-70">
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-rh-gray-40">
              <span>TTS: <span className="text-rh-gray-20">{ttsStreamStatus}</span></span>
              <span>Buffered: <span className="text-rh-gray-20">{ttsStreamBufferedMs.toFixed(0)}ms</span></span>
              <span>Min/Max: <span className="text-rh-gray-20">{ttsStreamMinBufferedMs.toFixed(0)}/{ttsStreamMaxBufferedMs.toFixed(0)}ms</span></span>
              <span>Underruns: <span className="text-rh-gray-20">{ttsStreamUnderruns}</span></span>
              <span>Rebuffers: <span className="text-rh-gray-20">{ttsStreamRebuffers}</span></span>
              <span>Gen: <span className="text-rh-gray-20">{ttsGenRealtimeX.toFixed(2)}x</span></span>
              <span>TTFT: <span className="text-rh-gray-20">{ttsTtftMs ? `${ttsTtftMs}ms` : "-"}</span></span>
              <span>TTFB: <span className="text-rh-gray-20">{ttsTtfbMs ? `${ttsTtfbMs}ms` : "-"}</span></span>
              <span>Chunks: <span className="text-rh-gray-20">{ttsStreamChunks}</span></span>
              <span>Bytes: <span className="text-rh-gray-20">{ttsStreamBytes}</span></span>
              <span>outHz: <span className="text-rh-gray-20">{ttsOutSampleRate || "-"}</span></span>
              {ttsRecordedUrl && (
                <>
                  <span>Recorded: <span className="text-rh-gray-20">{(ttsRecordedDurationMs / 1000).toFixed(2)}s</span></span>
                  <a href={ttsRecordedUrl} download={ttsRecordedFilename} className="text-rh-red hover:underline">Download WAV</a>
                </>
              )}
              <button onClick={() => stopTtsStream({ resetStats: true })} className="text-rh-red hover:underline">Stop Playback</button>
            </div>
          </div>
        </div>
      )}

      {/* ─── Main Content ────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">

        {/* ─── Left: Pizza + Action Buttons ──────────────────────────── */}
        <div className="flex-none lg:w-[420px] flex flex-col items-center justify-center p-6 lg:p-10 gap-6">
          {/* Pizza character */}
          <div className="relative">
            <TalkingPizza
              isTalking={isSpeaking}
              className="w-48 h-48 lg:w-64 lg:h-64 drop-shadow-2xl"
            />
            {isSpeaking && (
              <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 flex gap-1">
                {[0, 1, 2].map((i) => (
                  <div
                    key={i}
                    className="w-1.5 h-1.5 rounded-full bg-rh-red"
                    style={{
                      animation: `pulse-red 1s ${i * 0.2}s infinite`,
                    }}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Agent transcript bubble */}
          {lastAgentMessage && (
            <div className="max-w-xs text-center bg-rh-gray-90 border border-rh-gray-70 rounded-2xl px-4 py-3 text-sm text-rh-gray-20 animate-fade-in-up relative">
              <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-4 h-4 bg-rh-gray-90 border-l border-t border-rh-gray-70 rotate-45" />
              <span className="relative">{lastAgentMessage.content}</span>
            </div>
          )}

          {/* Status line */}
          <div className="text-xs text-rh-gray-50">
            {status === "recording" ? (
              <span className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-rh-red animate-pulse-red" />
                Recording...
              </span>
            ) : (
              status
            )}
          </div>

          {/* Big action buttons */}
          <div className="flex gap-4">
            <button
              className="rounded-xl px-8 py-4 text-lg font-bold transition-all disabled:opacity-30
                bg-rh-green text-white hover:bg-rh-green-dark
                shadow-lg hover:shadow-xl active:scale-95"
              style={{ minWidth: 120 }}
              onClick={() => {
                void primeTtsAudio();
                startRecording();
              }}
              disabled={!connected || status === "recording"}
            >
              TALK
            </button>
            <button
              className="rounded-xl px-8 py-4 text-lg font-bold transition-all disabled:opacity-30
                bg-rh-red text-white hover:bg-rh-red-dark
                shadow-lg hover:shadow-xl active:scale-95"
              style={{ minWidth: 120 }}
              onClick={stopAndSend}
              disabled={!connected || status !== "recording"}
            >
              SEND
            </button>
          </div>

          {/* Text input for typing */}
          <div className="w-full max-w-xs flex gap-2">
            <input
              className="flex-1 rounded-lg bg-rh-gray-90 border border-rh-gray-70 px-3 py-2 text-sm text-white placeholder-rh-gray-50 focus:border-rh-red focus:ring-1 focus:ring-rh-red focus:outline-none"
              placeholder="Type a message..."
              value={textToSend}
              onChange={(e) => setTextToSend(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendText();
                }
              }}
            />
            <button
              className="rounded-lg bg-rh-red text-white px-4 py-2 text-sm font-semibold hover:bg-rh-red-dark disabled:opacity-40 transition-colors"
              onClick={sendText}
              disabled={!connected}
            >
              Send
            </button>
          </div>

          {/* Error display */}
          {error && (
            <div className="w-full max-w-xs rounded-lg border border-red-900 bg-rh-red-bg/60 px-3 py-2 text-xs text-red-200">
              {error}
            </div>
          )}
        </div>

        {/* ─── Right: Transcript + Conversation ──────────────────────── */}
        <div className="flex-1 flex flex-col overflow-hidden border-l border-rh-gray-80">

          {/* Agent Transcript */}
          <div className="flex-none border-b border-rh-gray-80">
            <div className="px-6 py-3 border-b border-rh-gray-80 flex items-center justify-between">
              <h2 className="text-sm font-semibold" style={{ fontFamily: "'Red Hat Display', sans-serif" }}>
                Agent Transcript
              </h2>
              {pizzaType && (
                <span className="text-xs px-2 py-0.5 rounded bg-rh-red/20 text-rh-red border border-rh-red/30">
                  {pizzaType}
                </span>
              )}
            </div>
            <div className="px-6 py-3 max-h-32 overflow-y-auto rh-scroll">
              <p className="text-sm text-rh-gray-20 whitespace-pre-wrap">
                {transcript || "Waiting for speech..."}
              </p>
            </div>
          </div>

          {/* Conversation History */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="flex-none px-6 py-3 border-b border-rh-gray-80">
              <h2 className="text-sm font-semibold" style={{ fontFamily: "'Red Hat Display', sans-serif" }}>
                Conversation History
              </h2>
            </div>
            <div className="flex-1 overflow-y-auto rh-scroll px-6 py-4 space-y-3">
              {messages.length ? (
                messages.filter((m) => m.role !== "interrupt").map((m, i) => {
                  const isHuman = m.role === "human";
                  const isRouting = m.content.startsWith("Routing to");
                  if (isRouting) {
                    return (
                      <div key={i} className="flex justify-center">
                        <span className="text-xs text-rh-gray-50 bg-rh-gray-90 px-3 py-1 rounded-full">
                          {m.content}
                        </span>
                      </div>
                    );
                  }
                  return (
                    <div key={i} className={`flex ${isHuman ? "justify-end" : "justify-start"}`}>
                      <div
                        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                          isHuman
                            ? "bg-rh-red/15 border border-rh-red/20 text-rh-gray-10"
                            : "bg-rh-gray-90 border border-rh-gray-70 text-rh-gray-20"
                        }`}
                      >
                        <div className="text-[10px] uppercase tracking-wider text-rh-gray-50 mb-1 font-semibold">
                          {m.role}
                        </div>
                        <div className="text-sm whitespace-pre-wrap">{m.content}</div>
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="flex-1 flex items-center justify-center text-rh-gray-50 text-sm">
                  Say something to start a conversation...
                </div>
              )}
              {waiting && (
                <div className="flex justify-start">
                  <div className="bg-rh-gray-90 border border-rh-gray-70 rounded-2xl px-5 py-3 flex items-center gap-2">
                    <svg className="w-4 h-4 animate-spin text-rh-red" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    <span className="text-sm text-rh-gray-40">Thinking...</span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>
        </div>
      </div>

      {/* ─── Red Hat gradient divider at bottom ──────────────────────── */}
      <div className="flex-none h-px bg-gradient-to-r from-rh-red via-rh-red/20 to-transparent" />
    </div>
  );
}
