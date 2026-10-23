"""Listen to a Train_0.wav segment while watching its H_event_sum frames.

This is a new, standalone helper. It reads the project's WAV and W_standard.npy
but does not change or create files in the AcousticVirtualSensor repository.

Run with the Python environment already used by the project:
    python listen_event_pulse.py

Optional paths:
    python listen_event_pulse.py --wav "...\\Train_0.wav" --dictionary "...\\W_standard.npy"
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import webbrowser

import numpy as np
import soundfile as sf
from scipy.optimize import nnls
from scipy.signal import get_window


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAV = PROJECT_ROOT / "data" / "train" / "Train_0.wav"
DEFAULT_DICTIONARY = PROJECT_ROOT / "models" / "nmf" / "W_standard.npy"
SAMPLE_RATE = 44_100
N_FFT = 2_048
HOP_LENGTH = 1_024
EVENT_COMPONENTS = 24
NNLS_MAX_ITER = 20_000


def make_event_envelope(wav_path: Path, dictionary_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Run the project's audio -> STFT -> log magnitude -> NNLS -> H_event_sum path."""
    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Cần WAV {SAMPLE_RATE} Hz; file này là {sample_rate} Hz.")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1:
        raise ValueError(f"Không hỗ trợ WAV có dạng dữ liệu {audio.shape}.")
    audio = np.ascontiguousarray(audio - np.mean(audio), dtype=np.float32)
    if len(audio) < N_FFT:
        raise ValueError(f"WAV phải dài ít nhất {N_FFT} mẫu để tạo một frame STFT.")

    dictionary = np.load(dictionary_path, allow_pickle=False)
    if dictionary.ndim != 2 or dictionary.shape[0] != N_FFT // 2 + 1:
        raise ValueError(f"W_standard cần có {N_FFT // 2 + 1} hàng; nhận được {dictionary.shape}.")
    if dictionary.shape[1] < EVENT_COMPONENTS:
        raise ValueError(f"W_standard cần ít nhất {EVENT_COMPONENTS} cột event.")
    dictionary = np.ascontiguousarray(dictionary, dtype=np.float64)

    window = get_window("hann", N_FFT, fftbins=True).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(audio, N_FFT)[::HOP_LENGTH]
    spectrum = np.fft.rfft(frames * window, n=N_FFT, axis=1).T.astype(np.complex64)
    v_matrix = np.ascontiguousarray(np.log1p(np.abs(spectrum)), dtype=np.float32)

    activations = np.zeros((dictionary.shape[1], v_matrix.shape[1]), dtype=np.float32)
    for frame_index in range(v_matrix.shape[1]):
        h, _ = nnls(dictionary, v_matrix[:, frame_index], maxiter=NNLS_MAX_ITER)
        activations[:, frame_index] = h
    h_event_sum = np.sum(activations[:EVENT_COMPONENTS], axis=0, dtype=np.float32)
    return audio, np.ascontiguousarray(h_event_sum, dtype=np.float32)


def make_page() -> str:
    return r'''<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nghe xung H_event_sum</title>
<style>
:root{color-scheme:light dark;font-family:Segoe UI,Arial,sans-serif;background:Canvas;color:CanvasText}
*{box-sizing:border-box}body{margin:0 auto;padding:18px;max-width:1100px}
h1{font-size:1.35rem;font-weight:600;margin:0 0 14px}p{margin:5px 0 14px;color:GrayText}
.controls{display:flex;align-items:end;gap:10px;flex-wrap:wrap;margin:12px 0}
label{display:grid;gap:4px;font-size:.9rem}input,button{font:inherit;padding:8px 10px;border:1px solid GrayText;border-radius:5px;background:Canvas;color:CanvasText}
input{width:115px}button{cursor:pointer}button:disabled{opacity:.5;cursor:default}
.plot{width:100%;height:270px;display:block;touch-action:manipulation}#zoom{height:230px}
.status{min-height:1.4em;color:GrayText;font-size:.9rem}.legend{font-size:.85rem;color:GrayText}
audio{width:100%;margin:4px 0 8px}
</style>
</head>
<body>
<h1>H_event_sum và âm thanh của cùng đoạn frame</h1>
<p id="meta">Đang nạp đặc trưng…</p>
<div class="controls">
  <label>Frame bắt đầu A<input id="start" type="number" min="0" step="1"></label>
  <label>Frame kết thúc B<input id="end" type="number" min="0" step="1"></label>
  <button id="load">Chọn đoạn A–B</button>
  <button id="play" disabled>▶ Phát</button>
  <button id="stop" disabled>■ Dừng</button>
</div>
<div class="legend">Toàn cảnh: H_event_sum của toàn bộ Train_0.wav. Bấm hai lần để chọn frame A rồi B.</div>
<canvas id="event" class="plot" aria-label="Toàn bộ H_event_sum của Train_0.wav theo frame"></canvas>
<div class="legend">Phóng to: cùng H_event_sum, chỉ trong khoảng frame A–B. Vạch đỏ chạy theo âm thanh đang phát.</div>
<canvas id="zoom" class="plot" aria-label="Đoạn H_event_sum từ frame A đến frame B được phóng to"></canvas>
<audio id="audio" controls></audio>
<div id="status" class="status" aria-live="polite"></div>
<script>
const $=id=>document.getElementById(id), eventCanvas=$('event'), zoomCanvas=$('zoom');
const ec=eventCanvas.getContext('2d'), zc=zoomCanvas.getContext('2d'), player=$('audio');
let envelope=[], sampleRate=44100, hop=1024, totalSamples=0, maxFrame=0;
let clip={a:0,b:0,hEvent:[]}, choosingA=true, raf=0, playing=false;
function sizeCanvas(canvas,ctx){const dpr=window.devicePixelRatio||1,r=canvas.getBoundingClientRect();canvas.width=Math.round(r.width*dpr);canvas.height=Math.round(r.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);return {w:r.width,h:r.height};}
function axes(ctx,w,h,left,bottom,yTitle,xTitle){ctx.strokeStyle='color-mix(in srgb, CanvasText 18%, transparent)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,12);ctx.lineTo(left,h-bottom);ctx.lineTo(w-12,h-bottom);ctx.stroke();ctx.fillStyle='CanvasText';ctx.font='12px Segoe UI,Arial';ctx.fillText(yTitle,8,16);ctx.fillText(xTitle,w-65,h-5);}
function drawEvent(cursor=null){const {w,h}=sizeCanvas(eventCanvas,ec),L=64,R=16,T=20,B=34, pw=w-L-R,ph=h-T-B;if(!envelope.length)return;
 ec.clearRect(0,0,w,h);axes(ec,w,h,L,B,'H_event_sum','Frame');
 const ymax=Math.max(...envelope,1e-9)*1.08,x=i=>L+i/Math.max(1,envelope.length-1)*pw,y=v=>T+ph-v/ymax*ph;
 ec.fillStyle='color-mix(in srgb, #ef9d35 20%, transparent)';const aa=Math.min(clip.a,clip.b),bb=Math.max(clip.a,clip.b);ec.fillRect(x(aa),T,Math.max(1,x(bb)-x(aa)),ph);
 ec.strokeStyle='#2783c5';ec.lineWidth=1.5;ec.beginPath();for(let i=0;i<envelope.length;i++){const px=x(i),py=y(envelope[i]);if(i===0)ec.moveTo(px,py);else ec.lineTo(px,py)}ec.stroke();
 ec.fillStyle='CanvasText';ec.font='11px Segoe UI,Arial';for(let k=0;k<=4;k++){const v=ymax*k/4,py=y(v);ec.fillStyle='color-mix(in srgb, CanvasText 12%, transparent)';ec.beginPath();ec.moveTo(L,py);ec.lineTo(w-R,py);ec.stroke();ec.fillStyle='CanvasText';ec.fillText(v.toPrecision(2),4,py+4)}
 for(let k=0;k<=4;k++){const f=Math.round(maxFrame*k/4),px=x(f);ec.fillText(String(f),px-8,h-13)}
 if(cursor!==null){const px=x(Math.max(0,Math.min(maxFrame,cursor)));ec.strokeStyle='#db3b35';ec.lineWidth=2;ec.beginPath();ec.moveTo(px,T);ec.lineTo(px,T+ph);ec.stroke()}
}
function drawZoom(cursorSeconds=null){const {w,h}=sizeCanvas(zoomCanvas,zc),L=64,R=16,T=20,B=34,pw=w-L-R,ph=h-T-B;zc.clearRect(0,0,w,h);axes(zc,w,h,L,B,'H_event_sum','Thời gian (s)');
 if(!clip.hEvent.length){zc.fillStyle='GrayText';zc.fillText('Chọn khoảng frame A–B để phóng to H_event_sum',L+8,T+24);return}
 const ymax=Math.max(...clip.hEvent,1e-9)*1.08,duration=clip.duration;
 const xx=i=>L+(i/Math.max(1,clip.hEvent.length))*pw, yy=v=>T+ph-v/ymax*ph;
 zc.strokeStyle='color-mix(in srgb, CanvasText 14%, transparent)';zc.beginPath();zc.moveTo(L,T+ph);zc.lineTo(w-R,T+ph);zc.stroke();
 zc.strokeStyle='#2783c5';zc.lineWidth=2;zc.beginPath();for(let i=0;i<clip.hEvent.length;i++){const px=xx(i),py=yy(clip.hEvent[i]);if(i===0)zc.moveTo(px,py);else zc.lineTo(px,py)}zc.stroke();
 for(let k=0;k<=4;k++){const t=duration*k/4,px=L+t/duration*pw;zc.fillStyle='CanvasText';zc.font='11px Segoe UI,Arial';zc.fillText(t.toFixed(2),px-10,h-13)}
 if(cursorSeconds!==null){const px=L+Math.max(0,Math.min(duration,cursorSeconds))/duration*pw;zc.strokeStyle='#db3b35';zc.lineWidth=2;zc.beginPath();zc.moveTo(px,T);zc.lineTo(px,T+ph);zc.stroke()}
}
function sync(){if(!playing)return;const sec=player.currentTime,frame=clip.a+sec*sampleRate/hop;drawEvent(frame);drawZoom(sec);raf=requestAnimationFrame(sync)}
async function loadClip(){let a=Number($('start').value),b=Number($('end').value);a=Math.max(0,Math.min(maxFrame,Math.floor(a)));b=Math.max(0,Math.min(maxFrame,Math.floor(b)));if(a>b)[a,b]=[b,a];$('start').value=a;$('end').value=b;clip.a=a;clip.b=b;drawEvent();$('status').textContent='Đang lấy đoạn H_event_sum và âm thanh…';$('play').disabled=true;
 try{const r=await fetch(`/api/clip?a=${a}&b=${b}`);if(!r.ok)throw new Error(await r.text());const d=await r.json();clip.hEvent=d.h_event_sum;clip.duration=d.duration;player.src='data:audio/wav;base64,'+d.wav;player.load();drawZoom();$('play').disabled=false;$('status').textContent=`Đoạn ${a}–${b}: ${(d.duration).toFixed(3)} giây. Hai biểu đồ đều là H_event_sum; vạch đỏ chạy theo âm thanh.`}catch(e){$('status').textContent='Không nạp được đoạn âm thanh: '+e.message}}
$('load').addEventListener('click',loadClip);
$('play').addEventListener('click',async()=>{if(!player.src)await loadClip();player.currentTime=0;try{await player.play();playing=true;$('stop').disabled=false;cancelAnimationFrame(raf);sync()}catch(e){$('status').textContent='Trình duyệt không phát được âm thanh: '+e.message}});
$('stop').addEventListener('click',()=>{player.pause();playing=false;cancelAnimationFrame(raf);$('stop').disabled=true;drawEvent(clip.a+player.currentTime*sampleRate/hop);drawZoom(player.currentTime)});
player.addEventListener('ended',()=>{playing=false;cancelAnimationFrame(raf);$('stop').disabled=true;drawEvent(clip.b+1);drawZoom(clip.duration)});
eventCanvas.addEventListener('click',e=>{const r=eventCanvas.getBoundingClientRect(),L=64,R=16,ratio=Math.max(0,Math.min(1,(e.clientX-r.left-L)/(r.width-L-R))),f=Math.round(ratio*maxFrame);if(choosingA){$('start').value=f;$('end').value=f;choosingA=false;$('status').textContent='Đã chọn frame A. Bấm vị trí kết thúc B trên biểu đồ.'}else{$('end').value=f;choosingA=true;loadClip()}});
window.addEventListener('resize',()=>{drawEvent(playing?clip.a+player.currentTime*sampleRate/hop:null);drawZoom(playing?player.currentTime:null)});
fetch('/api/data').then(r=>{if(!r.ok)throw new Error('Không đọc được dữ liệu');return r.json()}).then(d=>{envelope=d.envelope;sampleRate=d.sample_rate;hop=d.hop;maxFrame=envelope.length-1;totalSamples=d.samples;$('start').value=d.default_a;$('end').value=d.default_b;$('meta').textContent=`${d.filename} · ${d.sample_rate.toLocaleString()} Hz · ${d.duration.toFixed(2)} giây · ${envelope.length} frame; khoảng cách frame ${(d.hop/d.sample_rate*1000).toFixed(2)} ms.`;loadClip()}).catch(e=>{$('meta').textContent='Lỗi: '+e.message});
</script>
</body>
</html>'''


def make_clip(audio: np.ndarray, envelope: np.ndarray, a: int, b: int) -> dict[str, object]:
    start_sample = a * HOP_LENGTH
    end_sample = min(len(audio), (b + 1) * HOP_LENGTH)
    segment = np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)
    if not len(segment):
        raise ValueError("Khoảng frame không có mẫu âm thanh.")

    pcm = np.clip(np.rint(segment * 32767.0), -32768, 32767).astype("<i2")
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())

    return {
        "wav": base64.b64encode(wav_buffer.getvalue()).decode("ascii"),
        "h_event_sum": envelope[a:b + 1].astype(float).tolist(),
        "duration": len(segment) / SAMPLE_RATE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Nghe và quan sát một đoạn xung H_event_sum của Train_0.wav")
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV, help="Đường dẫn WAV, mặc định là Train_0.wav")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY, help="Đường dẫn W_standard.npy")
    parser.add_argument("--port", type=int, default=8765, help="Cổng web nội bộ (mặc định 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Không tự mở trình duyệt")
    args = parser.parse_args()
    wav_path, dictionary_path = args.wav.expanduser().resolve(), args.dictionary.expanduser().resolve()
    if not wav_path.is_file():
        parser.error(f"Không tìm thấy WAV: {wav_path}")
    if not dictionary_path.is_file():
        parser.error(f"Không tìm thấy W_standard: {dictionary_path}")

    print(f"Đang tính H_event_sum từ {wav_path.name} …", flush=True)
    audio, envelope = make_event_envelope(wav_path, dictionary_path)
    peak = int(np.argmax(envelope))
    default_a = max(0, peak - 12)
    default_b = min(len(envelope) - 1, peak + 12)
    metadata = {
        "filename": wav_path.name,
        "sample_rate": SAMPLE_RATE,
        "hop": HOP_LENGTH,
        "samples": len(audio),
        "duration": len(audio) / SAMPLE_RATE,
        "envelope": envelope.tolist(),
        "default_a": default_a,
        "default_b": default_b,
    }
    page = make_page()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlparse(self.path)
            if route.path == "/":
                payload = page.encode("utf-8")
                content_type = "text/html; charset=utf-8"
            elif route.path == "/api/data":
                payload = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            elif route.path == "/api/clip":
                try:
                    query = parse_qs(route.query)
                    a = int(query.get("a", [default_a])[0])
                    b = int(query.get("b", [default_b])[0])
                    a, b = sorted((max(0, min(len(envelope) - 1, a)), max(0, min(len(envelope) - 1, b))))
                    payload = json.dumps(make_clip(audio, envelope, a, b), separators=(",", ":")).encode("utf-8")
                    content_type = "application/json; charset=utf-8"
                except (ValueError, TypeError) as exc:
                    self.send_error(400, str(exc))
                    return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *values: object) -> None:
            return

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        raise SystemExit(f"Không mở được cổng {args.port}: {exc}") from exc
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Đã tính {len(envelope)} frame; biên độ H lớn nhất ở frame {peak}.")
    print(f"Mở {url} để chọn frame và nghe. Nhấn Ctrl+C trong cửa sổ này để dừng.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng máy chủ nghe xung.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
