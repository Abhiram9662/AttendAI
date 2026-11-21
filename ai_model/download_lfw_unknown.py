
from pathlib import Path
import csv
import random
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "dataset" / "unknown_lfw"
IMAGE_DIR = OUTPUT_DIR / "images"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

TOTAL_IMAGES = 1000
RANDOM_SEED = 42

# --------------------------------------------------
# SAFETY: DO NOT TOUCH STUDENT DATA OR MODEL
# --------------------------------------------------

STUDENT_DIR = BASE_DIR / "dataset" / "raw"
MODEL_FILE = BASE_DIR / "trained_models" / "face_classifier.joblib"

if not STUDENT_DIR.exists():
    raise FileNotFoundError(
        f"Student dataset folder not found: {STUDENT_DIR}"
    )

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Existing trained model not found: {MODEL_FILE}"
    )

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Refuse to overwrite an existing LFW collection.
existing_images = list(IMAGE_DIR.glob("*.jpg"))

if existing_images:
    raise RuntimeError(
        "LFW images already exist. Nothing was overwritten.\n"
        "To download again, manually move or rename the existing "
        "unknown_lfw folder first."
    )

# --------------------------------------------------
# DOWNLOAD LFW
# --------------------------------------------------

print("Downloading/loading LFW dataset...")
print("This may take some time on the first run.")

lfw = fetch_lfw_people(
    data_home=str(BASE_DIR / "lfw_cache"),
    color=True,
    resize=1.0,
    min_faces_per_person=10,
    download_if_missing=True
)

print(f"Available images: {len(lfw.images)}")
print(f"Available identities: {len(lfw.target_names)}")

# --------------------------------------------------
# GROUP IMAGES BY LFW IDENTITY
# Identity names are not saved in the output.
# --------------------------------------------------

rng = random.Random(RANDOM_SEED)
groups = {}

for index, identity_id in enumerate(lfw.target):
    groups.setdefault(int(identity_id), []).append(index)

# Shuffle each identity's images.
eligible_groups = []

for identity_id, indices in groups.items():
    if len(indices) >= 2:
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        eligible_groups.append(shuffled)

rng.shuffle(eligible_groups)

if not eligible_groups:
    raise RuntimeError("No eligible LFW identities were found.")

# Round-robin selection avoids selecting all images
# from just a small number of identities.
selected = []
position = 0

while len(selected) < TOTAL_IMAGES:
    added_this_round = False

    for group in eligible_groups:
        if position < len(group):
            selected.append(group[position])
            added_this_round = True

            if len(selected) == TOTAL_IMAGES:
                break

    if not added_this_round:
        raise RuntimeError(
            f"Only {len(selected)} images were available."
        )

    position += 1

# --------------------------------------------------
# SAVE IMAGES AND MANIFEST
# --------------------------------------------------

manifest_rows = []

for number, dataset_index in enumerate(selected, start=1):
    image = lfw.images[dataset_index]

    # LFW provides RGB images; OpenCV writes BGR.
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

    filename = f"unknown_{number:04d}.jpg"
    output_path = IMAGE_DIR / filename

    success = cv2.imwrite(
        str(output_path),
        image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    if not success:
        raise RuntimeError(f"Could not save {output_path}")

    manifest_rows.append({
        "filename": filename,
        "lfw_identity_group": int(lfw.target[dataset_index]),
        "dataset_index": int(dataset_index)
    })

with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "lfw_identity_group",
            "dataset_index"
        ]
    )
    writer.writeheader()
    writer.writerows(manifest_rows)

print("\nLFW download completed.")
print(f"Images saved: {len(manifest_rows)}")
print(f"Image folder: {IMAGE_DIR}")
print(f"Manifest: {MANIFEST_PATH}")

print("\nExisting student datasets were not modified.")
print("Existing trained model was not modified.")
