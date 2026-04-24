# backend/app.py
import os, json, numpy as np, uuid, random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from voice_model import get_embedding_file, cosine_similarity
from behavior import extract_behavior_features, compare_behavior

app = Flask(__name__, static_folder="../frontend")
CORS(app)

DATA_ROOT = "../User_Voice"
BACKEND_DATA = "./backend_data"  # relative to backend/
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(BACKEND_DATA, exist_ok=True)

# Load centroids and behavior templates (built earlier)
def load_profiles():
    cent_path = os.path.join(BACKEND_DATA, "centroids.npy")
    bp_path = os.path.join(BACKEND_DATA, "behavior_profiles.json")
    thr_path = os.path.join(BACKEND_DATA, "threshold.json")
    centroids = {}
    if os.path.exists(cent_path):
        centroids = np.load(cent_path, allow_pickle=True).item()
    beh_profiles = {}
    if os.path.exists(bp_path):
        beh_profiles = json.load(open(bp_path))
    threshold = 0.5
    if os.path.exists(thr_path):
        threshold = json.load(open(thr_path))["global_threshold"]
    return centroids, beh_profiles, threshold

centroids, beh_profiles, GLOBAL_THRESHOLD = load_profiles()

# question bank (10 prompts)
QUESTION_BANK = [
    "My voice is my password.",
    "Unlock the door.",
    "Count slowly from one to ten.",
    "Please say your full name.",
    "Open the system now.",
    "Artificial intelligence is the future.",
    "Say the name of your hometown.",
    "Say your favorite color.",
    "Count backwards from ten to one.",
    "Today is a beautiful day."
]

@app.route("/register/start", methods=["POST"])
def register_start():
    data = request.get_json()
    name = data.get("name", "Unknown")
    # generate user id
    existing = sorted([d for d in os.listdir(DATA_ROOT) if d.startswith("U")])
    idx = len(existing) + 1
    user_id = f"U{idx:03d}"
    os.makedirs(os.path.join(DATA_ROOT, user_id), exist_ok=True)
    os.makedirs(os.path.join(DATA_ROOT, user_id, "answers"), exist_ok=True)
    questions = random.sample(QUESTION_BANK, 10)
    meta = {"name": name, "questions": questions}
    with open(os.path.join(DATA_ROOT, user_id, "meta.json"), "w") as f:
        json.dump(meta, f)
    return jsonify({"user_id": user_id, "questions": questions})

@app.route("/register/submit", methods=["POST"])
def register_submit():
    user_id = request.form["user_id"]
    q_index = int(request.form["q_index"])  # 1..10
    file = request.files["audio"]
    path = os.path.join(DATA_ROOT, user_id, "answers", f"{q_index:02d}.wav")
    file.save(path)
    return jsonify({"status": "saved", "path": path})

@app.route("/register/finalize", methods=["POST"])
def register_finalize():
    user_id = request.json["user_id"]
    user_folder = os.path.join(DATA_ROOT, user_id, "answers")
    files = sorted([os.path.join(user_folder, f) for f in os.listdir(user_folder) if f.endswith(".wav")])
    if len(files) < 3:
        return jsonify({"error": "Need at least 3 recordings to finalize"}), 400

    # compute embeddings and save
    embeddings = np.stack([get_embedding_file(f) for f in files], axis=0)
    np.save(os.path.join(DATA_ROOT, user_id, "embeddings.npy"), embeddings)

    # behavioral profile
    behs = [extract_behavior_features(f) for f in files]
    avg_beh = {k: float(np.mean([b[k] for b in behs])) for k in behs[0].keys()}
    with open(os.path.join(DATA_ROOT, user_id, "behavior.json"), "w") as f:
        json.dump(avg_beh, f)

    # update global backend profiles (append new centroid)
    centroid = np.mean(embeddings, axis=0); centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
    centroids[user_id] = centroid
    beh_profiles[user_id] = avg_beh
    # save back to BACKEND_DATA
    np.save(os.path.join(BACKEND_DATA, "centroids.npy"), centroids, allow_pickle=True)
    with open(os.path.join(BACKEND_DATA, "behavior_profiles.json"), "w") as f:
        json.dump(beh_profiles, f)
    with open(os.path.join(BACKEND_DATA, "threshold.json"), "w") as f:
        json.dump({"global_threshold": GLOBAL_THRESHOLD}, f)
    return jsonify({"status": "finalized", "user_id": user_id})

@app.route("/login/start", methods=["POST"])
def login_start():
    user_id = request.json.get("user_id")
    # pick random question from that user's meta if exists else random global
    meta_path = os.path.join(DATA_ROOT, user_id, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        q = random.choice(meta["questions"])
    else:
        q = random.choice(QUESTION_BANK)
    return jsonify({"question": q})

@app.route("/login/verify", methods=["POST"])
def login_verify():
    user_id = request.form.get("user_id")
    audio = request.files["audio"]
    tmp = "temp_live.wav"
    audio.save(tmp)

    if user_id not in centroids:
        return jsonify({"error": "Unknown user"}), 404

    # voice score vs centroid
    test_emb = get_embedding_file(tmp)
    centroid = centroids[user_id]
    voice_score = cosine_similarity(test_emb, centroid)

    # behavior score
    stored_beh = beh_profiles.get(user_id, None)
    test_beh = extract_behavior_features(tmp)
    behavior_score = compare_behavior(stored_beh, test_beh) if stored_beh else 0.5

    final_score = 0.7 * voice_score + 0.3 * behavior_score
    ok = final_score > GLOBAL_THRESHOLD
    return jsonify({
        "voice_score": voice_score,
        "behavior_score": behavior_score,
        "final_score": final_score,
        "verified": bool(ok)
    })

# minimal static file serving for frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join("../frontend", path)):
        return send_from_directory("../frontend", path)
    return send_from_directory("../frontend", "login.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
