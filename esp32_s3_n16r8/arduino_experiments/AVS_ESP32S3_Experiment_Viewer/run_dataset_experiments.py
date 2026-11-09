#!/usr/bin/env python3
"""Stream every train/test WAV once; ESP32 runs the full C++ profile sweep per frame."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import serial
import soundfile as sf
from scipy.signal import find_peaks, resample_poly

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BLOCK = 512
PROFILES = ["legacy_nnls", "m64_nnls", "m128_nnls"] + [
    f"m64_fista{n}" for n in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
] + ["m128_fista512"]


def read_audio(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = np.mean(audio, axis=1, dtype=np.float32)
    if rate != 44100:
        from math import gcd
        d = gcd(int(rate), 44100)
        audio = resample_poly(audio, 44100 // d, int(rate) // d).astype(np.float32)
    if audio.size < 2048:
        raise ValueError(f"{path.name}: audio shorter than one FFT frame")
    return np.ascontiguousarray(audio, dtype=np.float32)


def read_line(port: serial.Serial, timeout: float = 60.0) -> str:
    deadline = time.monotonic() + timeout
    pending=bytearray()
    while time.monotonic() < deadline:
        raw=port.read_until(b"\n",size=8192)
        if raw:
            pending.extend(raw)
            if pending.endswith(b"\n"):
                line=pending.decode("ascii","replace").strip()
                if line:return line
                pending.clear()
    suffix=pending.decode("ascii","replace")[-120:]
    raise TimeoutError(f"No complete response line from ESP32. Partial data: {suffix!r}")


def parse_key_values(fields: list[str]) -> dict[str, str]:
    out = {}
    for field in fields:
        if "=" in field:
            key, value = field.split("=", 1)
            out[key] = value
    return out


def load_labels(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            out[name.strip().removesuffix(".wav") + ".wav"] = int(value.strip())
    return out


def tune_peaks(samples: list[dict]) -> dict:
    candidates = [0.0]
    for sample in samples:
        peaks, props = find_peaks(sample["env"], prominence=0, distance=1)
        candidates.extend(float(p) for p in props["prominences"] if p > 0 and np.isfinite(p))
    thresholds = np.unique(np.quantile(candidates, np.linspace(0, 1, 101)))
    best = None
    for prominence in thresholds:
        for distance in range(1, 16):
            errors = [abs(find_peaks(s["env"], prominence=float(prominence), distance=distance)[0].size - s["truth"])
                      for s in samples]
            score = (float(np.mean(errors)), distance, float(prominence))
            if best is None or score < best:
                best = score
    return {"train_mae": best[0], "distance": best[1], "prominence": best[2]}


def fista_python(w: np.ndarray, v: np.ndarray, steps: int) -> np.ndarray:
    w=np.asarray(w,dtype=np.float32);v=np.asarray(v,dtype=np.float32)
    gram=np.asarray(w.T@w,dtype=np.float32);rhs=np.asarray(w.T@v,dtype=np.float32)
    lipschitz=float(np.max(np.sum(gram,axis=1,dtype=np.float32)))
    x=np.zeros_like(rhs);y=np.zeros_like(rhs);scale=np.float32(1)
    step=np.float32(1/lipschitz)
    for _ in range(steps):
        next_scale=np.float32(.5)*(np.float32(1)+np.sqrt(np.float32(1)+np.float32(4)*scale*scale))
        beta=(scale-np.float32(1))/next_scale
        x_next=np.maximum(np.float32(0),y+step*(rhs-gram@y))
        y=x_next+beta*(x_next-x);x,scale=x_next,next_scale
    return np.ascontiguousarray(x,dtype=np.float32)


def h_metrics(cpp_h: np.ndarray, py_h: np.ndarray) -> dict:
    common=min(cpp_h.shape[1],py_h.shape[1]);a=cpp_h[:,:common];b=py_h[:,:common]
    env_a=np.sum(a[:24],axis=0,dtype=np.float32);env_b=np.sum(b[:24],axis=0,dtype=np.float32)
    corr=float(np.corrcoef(env_a,env_b)[0,1]) if common>1 and np.std(env_a)>0 and np.std(env_b)>0 else float("nan")
    return {"frames_cpp":cpp_h.shape[1],"frames_python":py_h.shape[1],
      "relative_h_l2_error":float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),1e-12)),
      "event_envelope_correlation":corr,"event_envelope_relative_l2_error":float(np.linalg.norm(env_a-env_b)/max(float(np.linalg.norm(env_b)),1e-12)),
      "max_abs_h_error":float(np.max(np.abs(a-b)))}


def scipy_nnls_matrix(w: np.ndarray, v: np.ndarray) -> np.ndarray:
    from scipy.optimize import nnls
    h=np.empty((w.shape[1],v.shape[1]),dtype=np.float32)
    for i in range(v.shape[1]):
        h[:,i],_=nnls(w,v[:,i],maxiter=2000)
    return h


def safe_write_row(writer, row: dict) -> None:
    writer.writerow(row)


def main() -> None:
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "arduino_dataset_experiments")
    parser.add_argument("--max-files", type=int, default=0, help="debug only; zero means every WAV in the selected split")
    parser.add_argument("--max-seconds", type=float, default=0, help="debug only; truncate each file and suppress label metrics")
    args = parser.parse_args()

    train_labels = load_labels(ROOT / "data" / "train" / "label.txt")
    test_labels = load_labels(ROOT / "data" / "test" / "label.txt")
    files: list[tuple[str, Path]] = []
    if args.split in ("all", "train"):
        files += [("train", p) for p in sorted((ROOT / "data" / "train").glob("*.wav"))]
    if args.split in ("all", "test"):
        files += [("test", p) for p in sorted((ROOT / "data" / "test").glob("*.wav"))]
    if args.max_files:
        files = files[:args.max_files]
    if not files:
        raise FileNotFoundError("No WAV files found in the selected dataset split.")

    args.output.mkdir(parents=True, exist_ok=True)
    activation_path = args.output / "cpp_activations.csv"
    feature_path = args.output / "cpp_features.csv"
    file_stats: list[dict] = []
    envelope_bank: dict[tuple[str, str], list[float]] = {}
    timing_bank: dict[tuple[str, str], list[int]] = {}
    feature_errors: list[dict] = []
    memory_rows: list[dict] = []
    python_models = None
    python_h_errors: list[dict] = []
    python_env_bank: dict[tuple[str,str],np.ndarray] = {}
    sweep_timing_bank: dict[str,list[int]] = {}
    sweep_wall_bank: dict[str,list[int]] = {}
    cpp_peaks: dict[tuple[str,str],dict] = {}
    thresholds_sent=False
    truncated_stems:set[str]=set()

    # Prepare the matching Python Mel frontends only for feature-parity metrics.
    try:
        from experiments.pipeline_k48_mel import mel_frontend
        with np.load(ROOT / "experiments" / "pipeline_k48_mel" / "models" / "W_standard_mel.npz", allow_pickle=False) as m:
            model64 = {"bank": m["mel_bank"], "ref": float(m["compressor_reference"]),
                       "input": str(m["mel_input"].item()), "compressor": str(m["compressor"].item()),"W":m["W"]}
        with np.load(ROOT / "experiments" / "pipeline_k48_mel" / "runs" / "k128_magnitude_log1p" / "W_standard_mel.npz", allow_pickle=False) as m:
            model128 = {"bank": m["mel_bank"], "ref": float(m["compressor_reference"]),
                        "input": str(m["mel_input"].item()), "compressor": str(m["compressor"].item()),"W":m["W"]}
        old_w=np.asarray(np.load(ROOT/"models"/"nmf"/"W_standard.npy"),dtype=np.float32)
        python_models = (mel_frontend, model64, model128, old_w)
    except (OSError, KeyError, ImportError) as error:
        print(f"Python feature parity unavailable: {error}")

    with activation_path.open("w", newline="", encoding="utf-8") as act_file, \
         feature_path.open("w", newline="", encoding="utf-8") as feat_file:
        act_fields = ["split", "file", "frame", "profile", "frontend_us", "solver_us", "total_us", "event_score"] + [f"h{i}" for i in range(48)]
        feat_writer = csv.writer(feat_file)
        act_writer = csv.DictWriter(act_file, fieldnames=act_fields)
        act_writer.writeheader(); feat_writer.writerow(["split", "file", "frame", "feature", "bin", "value"])

        with serial.Serial(args.port, args.baud, timeout=1, write_timeout=60) as port:
            time.sleep(2); port.reset_input_buffer(); port.write(b"PING\n"); port.flush()
            deadline = time.monotonic() + 30;board_ready=False
            while not board_ready:
                if time.monotonic() > deadline:
                    raise TimeoutError("ESP32 did not announce AVS_DATASET_READY on COM port.")
                msg = read_line(port, 5)
                if msg == "PONG" or msg.startswith("AVS_DATASET_READY"):
                    board_ready=True
                elif msg.startswith("MEM,"):
                    memory_rows.append(parse_key_values(msg.split(",")[1:]))
            port.write(b"ABORT\n");port.flush()
            while True:
                msg=read_line(port,5)
                if msg=="ABORT_ACK":break
                if msg.startswith("MEM,"):memory_rows.append(parse_key_values(msg.split(",")[1:]))

            for file_no, (split, wav_path) in enumerate(files, 1):
                if split=="test" and not thresholds_sent:
                    if args.split=="all" or args.split=="test":
                        saved_params={}
                        params_path=ROOT/"experiments"/"pipeline_k48_mel"/"results"/"fista_peak_params.json"
                        if params_path.exists():
                            import json
                            saved_params=json.loads(params_path.read_text(encoding="utf-8"))
                        old_params_path=ROOT/"models"/"peak_detection"/"peak_params.json"
                        old_params=json.loads(old_params_path.read_text(encoding="utf-8")) if old_params_path.exists() else {"prominence":0.01,"distance":4}
                        for profile in PROFILES:
                            samples=[]
                            for name,truth_value in sorted(train_labels.items()):
                                env=envelope_bank.get((Path(name).stem,profile))
                                if env is not None:samples.append({"env":np.asarray(env),"truth":truth_value,"file":name})
                            if samples:params=tune_peaks(samples)
                            elif profile=="legacy_nnls":params=old_params
                            else:
                                key="python_scipy_nnls" if profile.endswith("nnls") else "fista_"+profile.removeprefix("m64_fista")
                                if profile=="m128_fista512":key="fista_512"
                                params=saved_params.get(key,saved_params.get("fista_512",{"prominence":0,"distance":1}))
                            port.write(f"THRESH {profile} {float(params['prominence']):.9g} {int(params['distance'])}\n".encode());port.flush()
                            response=read_line(port,10)
                            if not response.startswith(f"THRESH_ACK,{profile},"):
                                raise RuntimeError(f"ESP32 rejected peak parameters: {response}")
                    thresholds_sent=True
                print(f"[{file_no}/{len(files)}] {split}: {wav_path.name}", flush=True)
                audio = read_audio(wav_path)
                truth = (train_labels if split == "train" else test_labels).get(wav_path.name, -1)
                if args.max_seconds>0:
                    audio=audio[:max(2048,int(args.max_seconds*44100))]
                    truth=-1
                    truncated_stems.add(wav_path.stem)
                if split == "train":
                    metric_split = "train_validation" if truth >= 0 else "train_unlabeled"
                else:
                    metric_split = "test"
                mean = float(np.mean(audio, dtype=np.float64))
                slug = wav_path.stem.replace(" ", "_")
                port.write(f"FILE {slug} {audio.size} {mean:.10g} {truth} {metric_split}\n".encode()); port.flush()
                while True:
                    ack = read_line(port)
                    if ack.startswith("MEM,"):
                        memory_rows.append(parse_key_values(ack.split(",")[1:])); continue
                    if ack=="PONG":continue
                    if not ack.startswith(f"FILE_ACK,{slug},"):
                        raise RuntimeError(f"ESP32 rejected file header: {ack}")
                    break

                cpp64: list[list[float]] = []
                cpp128: list[list[float]] = []
                cpp_h: dict[str,list[list[float]]] = {profile:[] for profile in PROFILES}
                file_envelopes: dict[str, list[float]] = {profile: [] for profile in PROFILES}
                file_times: dict[str, list[int]] = {profile: [] for profile in PROFILES}
                frames_seen: set[int] = set()
                frame_solver_us: dict[int,dict[str,int]] = {}
                frame_front_us: dict[int,int] = {}

                def collect(line: str) -> None:
                    if line.startswith("ACT,"):
                        parts = line.split(",")
                        # ACT,split,file,frame,profile,frontend_us,solver_us,event,H0..H47
                        if len(parts) != 56:
                            raise RuntimeError(f"Malformed ACT record ({len(parts)} fields): {line[:160]}")
                        _, row_split, row_file, frame, profile, front, solve, score, *h = parts
                        frame_i = int(frame); front_i=int(front); solve_i=int(solve); score_f=float(score)
                        activations = [float(x) for x in h]
                        safe_write_row(act_writer, {"split": row_split, "file": row_file, "frame": frame_i,
                            "profile": profile, "frontend_us": front_i, "solver_us": solve_i,
                            "total_us": front_i + solve_i, "event_score": score_f,
                            **{f"h{i}": value for i, value in enumerate(activations)}})
                        file_envelopes[profile].append(score_f); file_times[profile].append(front_i + solve_i)
                        timing_bank.setdefault((row_split,profile),[]).append(front_i+solve_i)
                        cpp_h[profile].append(activations)
                        frame_solver_us.setdefault(frame_i,{})[profile]=solve_i
                        frame_front_us[frame_i]=front_i
                        frames_seen.add(frame_i)
                    elif line.startswith("MEL64,") or line.startswith("MEL128,"):
                        parts=line.split(","); kind=parts[0]; frame_i=int(parts[2]); values=[float(x) for x in parts[3:]]
                        expected=64 if kind=="MEL64" else 128
                        if len(values)!=expected: raise RuntimeError(f"Malformed {kind} feature frame {frame_i}")
                        (cpp64 if kind=="MEL64" else cpp128).append(values)
                        for band,value in enumerate(values): feat_writer.writerow([metric_split,slug,frame_i,kind.lower(),band,value])
                    elif line.startswith("MEM,"):
                        memory_rows.append(parse_key_values(line.split(",")[1:]))
                    elif line.startswith("FRAME_WALL,"):
                        parts=line.split(",");values=parse_key_values(parts[3:])
                        sweep_wall_bank.setdefault(metric_split,[]).append(int(values.get("wall_us",0)))
                    elif line.startswith("PEAK,"):
                        parts=line.split(",")
                        if len(parts)<7:raise RuntimeError(f"Malformed C++ peak record: {line[:160]}")
                        _,row_split,row_file,profile,prominence,distance,count,*indices=parts
                        cpp_peaks[(row_file,profile)]={"split":row_split,"count":int(count),
                          "prominence":float(prominence),"distance":int(distance),"indices":[int(x) for x in indices]}
                    elif line.startswith("PEAK,"):
                        parts=line.split(",")
                        _,row_split,row_file,profile,prominence,distance,count,*indices=parts
                        cpp_peaks[(row_file,profile)]={"split":row_split,"count":int(count),
                            "prominence":float(prominence),"distance":int(distance),"indices":[int(x) for x in indices]}
                    elif line.startswith("ERROR,"):
                        raise RuntimeError(line)

                for start in range(0, audio.size, BLOCK):
                    chunk=np.asarray(audio[start:start+BLOCK],dtype="<f4")
                    port.write(f"DATA {chunk.size}\n".encode());port.write(chunk.tobytes());port.flush()
                    expected_ack=start+chunk.size
                    while True:
                        response=read_line(port, timeout=180)
                        if response.startswith("ACK,"):
                            if int(response.split(",",1)[1])!=expected_ack: raise RuntimeError(f"Sample ACK mismatch: {response}")
                            break
                        collect(response)
                port.write(b"END\n");port.flush()
                file_done=None;got_memory=False
                while file_done is None or not got_memory:
                    response=read_line(port, timeout=60)
                    if response.startswith("FILE_DONE,"):
                        file_done=response.split(",")
                    elif response.startswith("MEM,"):
                        memory_rows.append(parse_key_values(response.split(",")[1:]));got_memory=True
                    else:
                        collect(response)
                actual_frames=int(file_done[2]); expected_frames=0 if audio.size<2048 else 1+(audio.size-2048)//1024
                if actual_frames!=expected_frames or len(frames_seen)!=expected_frames:
                    raise RuntimeError(f"Frame loss/mismatch for {wav_path.name}: MCU={actual_frames}, received={len(frames_seen)}, expected={expected_frames}")
                unique_solver_profiles=("legacy_nnls","m64_nnls","m128_nnls","m64_fista4096","m128_fista512")
                for frame_i,profile_times in frame_solver_us.items():
                    solver_us=sum(profile_times.get(name,0) for name in unique_solver_profiles)
                    sweep_timing_bank.setdefault(metric_split,[]).append(frame_front_us[frame_i]+solver_us)
                file_stats.append({"split":metric_split,"file":wav_path.name,"truth":truth,"samples":audio.size,"frames":actual_frames})
                for profile in PROFILES:envelope_bank[(slug,profile)]=file_envelopes[profile]
                print(f"    {actual_frames} frames; MCU frame-count check passed",flush=True)

                if python_models is not None and cpp64 and cpp128:
                    frontend, m64, m128, old_w = python_models
                    py_values=[]
                    for model in (m64,m128):
                        mel=frontend.mel_spectrogram(audio,model["bank"],model["input"])
                        py_values.append(frontend.compress_mel(mel,model["ref"],model["compressor"]))
                    for label,cpp,py in (("mel64",cpp64,py_values[0]),("mel128",cpp128,py_values[1])):
                        cpp_arr=np.asarray(cpp,dtype=np.float32).T
                        py_arr=np.asarray(py,dtype=np.float32)
                        common=min(cpp_arr.shape[1],py_arr.shape[1])
                        diff=cpp_arr[:,:common]-py_arr[:,:common]
                        feature_errors.append({"split":metric_split,"file":wav_path.name,"feature":label,
                          "frames_cpp":cpp_arr.shape[1],"frames_python":py_arr.shape[1],
                          "relative_l2_error":float(np.linalg.norm(diff)/max(np.linalg.norm(py_arr[:,:common]),1e-12)),
                          "mae":float(np.mean(np.abs(diff))),"max_abs_error":float(np.max(np.abs(diff)))})
                    np.savez_compressed(args.output / f"{slug}_cpp_features.npz",mel64=np.asarray(cpp64,dtype=np.float32).T,mel128=np.asarray(cpp128,dtype=np.float32).T,
                                        mel64_python=py_values[0],mel128_python=py_values[1])
                    cpp_arrays={name:np.asarray(values,dtype=np.float32).T for name,values in cpp_h.items()}
                    mag=frontend.stft_magnitude(audio)
                    py_old=scipy_nnls_matrix(old_w,np.log1p(mag).astype(np.float32))
                    py64_nnls=scipy_nnls_matrix(np.asarray(m64["W"],dtype=np.float32),py_values[0])
                    py128_nnls=scipy_nnls_matrix(np.asarray(m128["W"],dtype=np.float32),py_values[1])
                    refs={"legacy_nnls":py_old,"m64_nnls":py64_nnls,"m128_nnls":py128_nnls}
                    for steps in (8,16,32,64,128,256,512,1024,2048,4096):
                        archive=ROOT/"experiments"/"pipeline_k48_mel"/"results"/f"activations_fista_{steps}.npz"
                        py_h=None
                        if archive.exists():
                            with np.load(archive,allow_pickle=False) as saved:
                                if wav_path.name in saved.files:py_h=np.asarray(saved[wav_path.name],dtype=np.float32)
                        if py_h is None:py_h=fista_python(np.asarray(m64["W"],dtype=np.float32),py_values[0],steps)
                        refs[f"m64_fista{steps}"]=py_h
                    refs["m128_fista512"]=fista_python(np.asarray(m128["W"],dtype=np.float32),py_values[1],512)
                    for profile,py_h in refs.items():
                        parity=h_metrics(cpp_arrays[profile],py_h)
                        python_h_errors.append({"split":metric_split,"file":wav_path.name,"profile":profile,**parity})
                        python_env_bank[(slug,profile)]=np.sum(py_h[:24],axis=0,dtype=np.float32)

    # Tune p/d only on labeled training files; apply frozen settings to test.
    prediction_rows=[]; metric_rows=[]; python_count_rows=[]
    labeled_train={p.name for p in (ROOT/"data"/"train").glob("*.wav") if p.name in train_labels}
    # Envelopes have been written to the activation CSV; load just score columns for metrics.
    all_env: dict[tuple[str,str], list[float]]={}
    with activation_path.open(encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            key=(row["file"],row["profile"]);all_env.setdefault(key,[]).append(float(row["event_score"]))
    truths={**train_labels,**test_labels}
    for profile in PROFILES:
        calibration=[]
        for name in sorted(labeled_train):
            stem=name.removesuffix(".wav")
            if stem in truncated_stems:continue
            env=all_env.get((stem,profile))
            if env is not None:calibration.append({"env":np.asarray(env),"truth":train_labels[name],"file":name})
        if calibration:
            params=tune_peaks(calibration)
        else:
            params_path=ROOT/"experiments"/"pipeline_k48_mel"/"results"/"fista_peak_params.json"
            if not params_path.exists():continue
            import json
            saved=json.loads(params_path.read_text(encoding="utf-8"))
            key="python_scipy_nnls" if profile.endswith("nnls") else "fista_"+profile.removeprefix("m64_fista")
            if profile=="m128_fista512":key="fista_512"
            if profile=="legacy_nnls":
                params=json.loads((ROOT/"models"/"peak_detection"/"peak_params.json").read_text(encoding="utf-8"))
            else:params=saved.get(key,saved.get("fista_512"))
        file_by_stem={p.stem:p for _,p in files}
        for (stem,model),env_values in all_env.items():
            if model!=profile or stem not in file_by_stem:continue
            path=file_by_stem[stem];split="train" if path.parent.name=="train" else "test"
            truth=-1 if stem in truncated_stems else truths.get(path.name,-1);env=np.asarray(env_values)
            peaks=find_peaks(env,prominence=params["prominence"],distance=params["distance"])[0]
            predicted=int(peaks.size)
            cpp_peak=cpp_peaks.get((stem,profile))
            prediction_rows.append({"split":split,"file":path.name,"profile":profile,"truth":truth,
              "prediction":predicted,"absolute_error":abs(predicted-truth) if truth>=0 else "",
                "cpp_peak_count":cpp_peak["count"] if cpp_peak else "",
                "cpp_peak_indices":";".join(map(str,cpp_peak["indices"])) if cpp_peak else "",
                "cpp_vs_python_peak_count_delta":cpp_peak["count"]-predicted if cpp_peak else "",
              "peak_indices":';'.join(map(str,peaks.tolist())),"prominence":params["prominence"],"distance":params["distance"]})
            py_env=python_env_bank.get((stem,profile))
            if py_env is not None:
                py_peaks=find_peaks(py_env,prominence=params["prominence"],distance=params["distance"])[0]
                python_count_rows.append({"split":split,"file":path.name,"profile":profile,"truth":truth,
                    "cpp_prediction":predicted,"python_prediction":int(py_peaks.size),
                    "prediction_difference":predicted-int(py_peaks.size),
                    "cpp_accuracy_percent":max(0.,1.-abs(predicted-truth)/max(1,truth))*100 if truth>=0 else "",
                    "python_accuracy_percent":max(0.,1.-abs(int(py_peaks.size)-truth)/max(1,truth))*100 if truth>=0 else ""})
            if truth>=0:
                accuracy=max(0.,1.-abs(predicted-truth)/max(1,truth))*100
                metric_rows.append({"split":split,"file":path.name,"profile":profile,"truth":truth,
                    "prediction":predicted,"absolute_error":abs(predicted-truth),"count_accuracy_percent":accuracy,
                    "count_tp":min(predicted,truth),"count_fp":max(predicted-truth,0),"count_fn":max(truth-predicted,0)})
        for split_name in ("train","test"):
            rows=[r for r in metric_rows if r["split"]==split_name and r["profile"]==profile]
            if not rows:continue
            tp=sum(r["count_tp"] for r in rows);fp=sum(r["count_fp"] for r in rows);fn=sum(r["count_fn"] for r in rows)
            metric_rows.append({"split":split_name,"file":"__SUMMARY__","profile":profile,"truth":sum(r["truth"] for r in rows),
                "prediction":sum(r["prediction"] for r in rows),"absolute_error":float(np.mean([r["absolute_error"] for r in rows])),
                "count_accuracy_percent":float(np.mean([r["count_accuracy_percent"] for r in rows])),
                "count_tp":tp,"count_fp":fp,"count_fn":fn,"count_f1":(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0),
                "peak_prominence":params["prominence"],"peak_distance":params["distance"],"tune_train_mae":params.get("train_mae","")})

    with (args.output/"predictions_per_file.csv").open("w",newline="",encoding="utf-8") as f:
        fields=sorted({k for r in prediction_rows for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(prediction_rows)
    with (args.output/"count_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        fields=sorted({k for r in metric_rows for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(metric_rows)
    with (args.output/"feature_parity.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["split","file","feature","frames_cpp","frames_python","relative_l2_error","mae","max_abs_error"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(feature_errors)
    with (args.output/"activation_parity.csv").open("w",newline="",encoding="utf-8") as f:
        fields=sorted({k for r in python_h_errors for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(python_h_errors)
    with (args.output/"python_cpp_count_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        fields=sorted({k for r in python_count_rows for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(python_count_rows)
    py_cpp_summary=[]
    for split_name in ("train","test"):
        for profile in PROFILES:
            rows=[r for r in python_count_rows if r["split"]==split_name and r["profile"]==profile and r["truth"]>=0]
            if rows:
                py_cpp_summary.append({"split":split_name,"profile":profile,"files":len(rows),
                  "cpp_mean_count_accuracy_percent":float(np.mean([r["cpp_accuracy_percent"] for r in rows])),
                  "python_mean_count_accuracy_percent":float(np.mean([r["python_accuracy_percent"] for r in rows])),
                  "mean_abs_count_difference_cpp_python":float(np.mean([abs(r["prediction_difference"]) for r in rows]))})
    with (args.output/"python_cpp_count_summary.csv").open("w",newline="",encoding="utf-8") as f:
        fields=sorted({k for r in py_cpp_summary for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(py_cpp_summary)
    with (args.output/"memory_samples.csv").open("w",newline="",encoding="utf-8") as f:
        fields=sorted({k for r in memory_rows for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(memory_rows)
    with (args.output/"files_processed.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["split","file","truth","samples","frames"];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(file_stats)

    # Summarize MCU compute latency per profile and split (frontend + solver, telemetry excluded).
    timing_rows=[]
    for split in ("train_calibration","train_validation","train_unlabeled","test"):
        for profile in PROFILES:
            vals=timing_bank.get((split,profile),[])
            if vals:
                ms=np.asarray(vals,dtype=float)/1000
                timing_rows.append({"split":split,"profile":profile,"frames":len(vals),"mean_ms":float(np.mean(ms)),
                    "p50_ms":float(np.percentile(ms,50)),"p95_ms":float(np.percentile(ms,95)),
                    "p99_ms":float(np.percentile(ms,99)),"max_ms":float(np.max(ms)),
                    "realtime_limit_ms":23.22,"frames_within_realtime_percent":float(np.mean(ms<=23.22)*100)})
    for split,vals in sweep_timing_bank.items():
        ms=np.asarray(vals,dtype=float)/1000
        timing_rows.append({"split":split,"profile":"ALL_PROFILES_ONE_PASS","frames":len(vals),"mean_ms":float(np.mean(ms)),
          "p50_ms":float(np.percentile(ms,50)),"p95_ms":float(np.percentile(ms,95)),"p99_ms":float(np.percentile(ms,99)),
          "max_ms":float(np.max(ms)),"realtime_limit_ms":23.22,"frames_within_realtime_percent":float(np.mean(ms<=23.22)*100)})
    for split,vals in sweep_wall_bank.items():
        ms=np.asarray(vals,dtype=float)/1000
        timing_rows.append({"split":split,"profile":"ALL_PROFILES_WITH_SERIAL_OUTPUT","frames":len(vals),"mean_ms":float(np.mean(ms)),
          "p50_ms":float(np.percentile(ms,50)),"p95_ms":float(np.percentile(ms,95)),"p99_ms":float(np.percentile(ms,99)),
          "max_ms":float(np.max(ms)),"realtime_limit_ms":23.22,"frames_within_realtime_percent":float(np.mean(ms<=23.22)*100)})
    with (args.output/"timing_summary.csv").open("w",newline="",encoding="utf-8") as f:
        fields=sorted({k for r in timing_rows for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(timing_rows)
    summaries=[r for r in metric_rows if r.get("file")=="__SUMMARY__"]
    if summaries:
        import matplotlib.pyplot as plt
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,10))
        x=np.arange(len(PROFILES));width=.36
        for offset,split_name,color in ((-width/2,"train","tab:blue"),(width/2,"test","tab:orange")):
            values={r["profile"]:r["count_accuracy_percent"] for r in summaries if r["split"]==split_name}
            ax1.bar(x+offset,[values.get(p,0) for p in PROFILES],width,label=split_name,color=color)
        ax1.set_xticks(x,PROFILES,rotation=35,ha="right");ax1.set_ylabel("Mean per-file count accuracy (%)")
        ax1.set_title("ESP32-S3 C++ profile comparison");ax1.grid(axis="y",alpha=.25);ax1.legend()
        for split_name,color in (("train_validation","tab:blue"),("test","tab:orange")):
            rows=[r for r in timing_rows if r["split"]==split_name and r["profile"] in PROFILES]
            mapping={r["profile"]:r["p95_ms"] for r in rows}
            ax2.plot(PROFILES,[mapping.get(p,np.nan) for p in PROFILES],marker="o",label=split_name,color=color)
        ax2.axhline(23.22,color="tab:red",linestyle="--",label="23.22 ms/frame")
        ax2.set_xticks(x,PROFILES,rotation=35,ha="right");ax2.set_ylabel("P95 compute time (ms/frame)")
        ax2.grid(alpha=.25);ax2.legend();fig.tight_layout();fig.savefig(args.output/"dataset_comparison.png",dpi=160)
    print(f"Saved dataset results to {args.output}")
    print("Count F1 is count-based (derived from per-file product counts); event timestamp labels are not present in label.txt.")
    print("Peak parameters are tuned using all labeled train WAVs and then frozen for test.")


if __name__ == "__main__":
    main()
