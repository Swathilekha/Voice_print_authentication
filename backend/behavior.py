# backend/behavior.py
import numpy as np
import librosa
import math

SR = 16000

def extract_behavior_features(file_path, sr=SR):
    """Return a dict of interpretable behavioral features."""
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) == 0:
        # blank file edge-case
        return {"pitch_mean":0,"pitch_std":0,"rms":0,"speaking_rate":0,"pause_ratio":1.0}

    # energy / RMS
    rms = float(np.mean(librosa.feature.rms(y=y)))

    # pitch via librosa YIN
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0_nonzero = f0[f0 > 0]
        pitch_mean = float(np.mean(f0_nonzero)) if len(f0_nonzero) else 0.0
        pitch_std  = float(np.std(f0_nonzero)) if len(f0_nonzero) else 0.0
    except Exception:
        pitch_mean, pitch_std = 0.0, 0.0

    # speaking rate approximation (onset counts / duration)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    speaking_rate = float(len(onsets) / duration) if duration > 0 else 0.0

    # pause ratio: fraction frames below threshold
    hop = 512
    frame_rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    thr = 0.2 * np.mean(frame_rms + 1e-9)
    pause_ratio = float(np.mean(frame_rms < thr))

    return {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "rms": rms,
        "speaking_rate": speaking_rate,
        "pause_ratio": pause_ratio
    }

def compare_behavior(profile_base, profile_test):
    """Return a behavior similarity score 0..1 (1 = identical)."""
    # Use normalized difference per feature and average
    keys = profile_base.keys()
    scores = []
    for k in keys:
        a = profile_base[k]
        b = profile_test[k]
        # handle zero cases
        denom = max(abs(a), abs(b), 1e-6)
        # similarity = 1 - relative difference (clipped at 0)
        sim = max(0.0, 1.0 - abs(a - b) / denom)
        scores.append(sim)
    return float(np.mean(scores))
