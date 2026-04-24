# backend/build_profiles.py
import os, json, numpy as np
from collections import defaultdict
from voice_model import get_embedding_file, cosine_similarity
from behavior import extract_behavior_features
from sklearn.metrics import roc_curve
import joblib

DATA_ROOT = os.path.join("..", "VoicePrint/User_Voice")
OUTPUT = os.path.join(os.path.dirname(__file__), "backend_data")

os.makedirs(OUTPUT, exist_ok=True)

def build():
    # collect embeddings per user
    embs = defaultdict(list)
    behs = defaultdict(list)
    users = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    for u in users:
        folder = os.path.join(DATA_ROOT, u)
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".wav")])
        for f in files:
            path = os.path.join(folder, f)
            try:
                e = get_embedding_file(path)
                embs[u].append(e)
                behs[u].append(extract_behavior_features(path))
            except Exception as ex:
                print("Error processing", path, ex)

    # centroid
    centroids = {}
    for u, L in embs.items():
        cent = np.mean(np.stack(L, axis=0), axis=0)
        cent /= (np.linalg.norm(cent) + 1e-10)
        centroids[u] = cent

    # compute global impostor/genuine distributions to choose threshold
    intra = []
    inter = []
    for u, L in embs.items():
        cent = centroids[u]
        for e in L:
            intra.append(float(np.dot(e, cent)))
            for v, c2 in centroids.items():
                if v == u: continue
                inter.append(float(np.dot(e, c2)))

    intra = np.array(intra); inter = np.array(inter)
    print("INTRA median:", np.median(intra), "INTER median:", np.median(inter))

    # default threshold (simple heuristic)
    thr = float(np.percentile(inter, 95) + 0.02)  # slightly above 95% impostor
    print("Suggested global threshold:", thr)

    # store centroids and behavior templates & per-user min genuine
    np.save(os.path.join(OUTPUT, "centroids.npy"), centroids, allow_pickle=True)
    # average behavior per user
    beh_profiles = {}
    for u, L in behs.items():
        # compute mean per key
        keys = L[0].keys()
        mean_profile = {k: float(np.mean([b[k] for b in L])) for k in keys}
        beh_profiles[u] = mean_profile

    json.dump(beh_profiles, open(os.path.join(OUTPUT, "behavior_profiles.json"), "w"))

    # store recommended threshold
    with open(os.path.join(OUTPUT, "threshold.json"), "w") as f:
        json.dump({"global_threshold": thr}, f)

    print("Saved centroids and behavior templates to", OUTPUT)

if __name__ == "__main__":
    build()
