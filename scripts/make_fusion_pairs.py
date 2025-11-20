
import os
import json
import random

def make_fusion_pairs(face_meta, audio_meta, out_json):
    pairs = []
    for f in face_meta:
        # match by label (live-live, spoof-spoof)
        same_label_audio = [a for a in audio_meta if a["label"] == f["label"]]
        if len(same_label_audio) == 0:
            continue
        a = random.choice(same_label_audio)
        pairs.append({
            "face_frames": f["frames_dir"],
            "audio_mel": a["mel_path"],
            "label": f["label"],
            "face_subject": f["subject"],
            "audio_speaker": a["speaker"]
        })
    with open(out_json, "w") as f:
        json.dump(pairs, f, indent=4)

