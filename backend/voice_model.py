# backend/voice_model.py
import os
import numpy as np
import torch
import torchaudio
import librosa
from speechbrain.pretrained import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

# PATH to the SpeechBrain model folder you downloaded earlier
SAVEDIR = os.path.join("..", "VoicePrint/pretrained_models", "speechbrain_ecapa_voxceleb")

device = "cuda" if torch.cuda.is_available() else "cpu"

spk_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=os.path.abspath(SAVEDIR),
    run_opts={"device": device},
    local_strategy=LocalStrategy.COPY  # 🧠 Force COPY instead of SYMLINK
)


def preprocess_audio_array(y, sr=16000, trim_db=30):
    y, _ = librosa.effects.trim(y, top_db=trim_db)
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)))
    y = y.astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-10)
    return y

def get_embedding_file(file_path, sr=16000):
    """Return L2-normalized 1D numpy embedding for file_path."""
    # Try SpeechBrain's encode_file (most robust)
    try:
        emb_t = spk_model.encode_file(file_path)
        emb = emb_t.squeeze().cpu().numpy()
    except Exception:
        # fallback: load waveform and encode_batch
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        y = preprocess_audio_array(y, sr=sr)
        tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
        emb_t = spk_model.encode_batch(tensor)
        emb = emb_t.squeeze().detach().cpu().numpy().flatten()
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))

def identify_by_centroid(test_file, user_centroids):
    emb = get_embedding_file(test_file)
    best_user, best_sim = None, -1.0
    for u, c in user_centroids.items():
        sim = cosine_similarity(emb, c)
        if sim > best_sim:
            best_sim, best_user = sim, u
    return best_user, best_sim
