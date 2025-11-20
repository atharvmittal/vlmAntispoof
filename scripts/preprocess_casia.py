
import os
import cv2
import json

def extract_frames_from_video(video_path, out_dir, max_frames=20):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total // max_frames, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            out_path = os.path.join(out_dir, f"{count:05d}.jpg")
            frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(out_path, frame)
        count += 1
    cap.release()


def preprocess_casia(base_raw, base_out):
    # Folders depend on final CASIA structure
    entries = []
    for root, dirs, files in os.walk(base_raw):
        for f in files:
            if f.lower().endswith(".avi"):
                video_path = os.path.join(root, f)
                label = "spoof" if "attack" in root.lower() else "live"
                subject = root.split("/")[-1]

                out_dir = os.path.join(base_out, label, subject, f.replace(".avi",""))
                extract_frames_from_video(video_path, out_dir)

                entries.append({
                    "video": video_path,
                    "label": label,
                    "subject": subject,
                    "frames_dir": out_dir
                })

    # save metadata
    os.makedirs(base_out, exist_ok=True)
    with open(os.path.join(base_out, "metadata.json"), "w") as f:
        json.dump(entries, f, indent=4)

