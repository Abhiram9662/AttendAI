
from pathlib import Path
import csv

import joblib
import numpy as np
import torch

from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


# ==================================================
# 1. PATHS AND CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

STUDENT_DIR = BASE_DIR / "dataset" / "raw"
UNKNOWN_DIR = BASE_DIR / "dataset" / "unknown_lfw" / "images"

MODEL_PATH = BASE_DIR / "trained_models" / "face_classifier.joblib"

RESULT_DIR = BASE_DIR / "evaluation_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULT_DIR / "lfw_unknown_results.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Exploratory thresholds only. Do not use these
# directly for live attendance decisions.
COSINE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]


# ==================================================
# 2. CHECK EXISTING FILES
# ==================================================

if not STUDENT_DIR.exists():
    raise FileNotFoundError(f"Missing: {STUDENT_DIR}")

if not UNKNOWN_DIR.exists():
    raise FileNotFoundError(f"Missing: {UNKNOWN_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing: {MODEL_PATH}")

print("=" * 60)
print("UNKNOWN PERSON EVALUATION - LFW")
print("=" * 60)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==================================================
# 3. LOAD EXISTING CLASSIFIER
# ==================================================

print("\nLoading existing SVM classifier...")

model_data = joblib.load(MODEL_PATH)

classifier = model_data["classifier"]

print("Loaded:", model_data.get("model_name", "SVM classifier"))
print("Registered students:", model_data.get("classes", []))


# ==================================================
# 4. INITIALIZE MTCNN AND FACENET
# ==================================================

print("\nLoading MTCNN...")

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    post_process=True,
    device=DEVICE
)

print("Loading FaceNet...")

facenet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(DEVICE)


# ==================================================
# 5. FACE EMBEDDING FUNCTION
# ==================================================

def extract_embedding(image_path):

    try:
        image = Image.open(image_path).convert("RGB")

        face = mtcnn(image)

        if face is None:
            return None

        face = face.unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            embedding = facenet(face)

        vector = embedding.cpu().numpy()[0]

        return vector.astype(np.float32)

    except Exception as error:
        print(f"Error processing {image_path.name}: {error}")
        return None


def normalize(vector):

    norm = np.linalg.norm(vector)

    if norm == 0:
        return None

    return vector / norm


# ==================================================
# 6. BUILD STUDENT REFERENCE EMBEDDINGS
# ==================================================

print("\nExtracting reference embeddings from students...")

student_centroids = {}

for student_folder in sorted(STUDENT_DIR.iterdir()):

    if not student_folder.is_dir():
        continue

    student_id = student_folder.name

    image_files = sorted([
        *student_folder.glob("*.jpg"),
        *student_folder.glob("*.jpeg"),
        *student_folder.glob("*.png")
    ])

    student_vectors = []

    print(f"\nStudent: {student_id}")
    print("Images:", len(image_files))

    for image_path in image_files:

        vector = extract_embedding(image_path)

        if vector is not None:
            normalized = normalize(vector)

            if normalized is not None:
                student_vectors.append(normalized)

    if not student_vectors:
        print("WARNING: No valid student embeddings.")
        continue

    centroid = np.mean(
        np.asarray(student_vectors),
        axis=0
    )

    centroid = normalize(centroid)

    student_centroids[student_id] = centroid

    print("Valid reference embeddings:", len(student_vectors))


if len(student_centroids) < 2:
    raise RuntimeError(
        "Could not build reference embeddings for both students."
    )


# ==================================================
# 7. LOAD UNKNOWN LFW IMAGES
# ==================================================

unknown_images = sorted([
    *UNKNOWN_DIR.glob("*.jpg"),
    *UNKNOWN_DIR.glob("*.jpeg"),
    *UNKNOWN_DIR.glob("*.png")
])

print("\nUnknown images found:", len(unknown_images))

if not unknown_images:
    raise RuntimeError("No LFW images found.")


# ==================================================
# 8. EVALUATE UNKNOWN IMAGES
# ==================================================

results = []

processed_count = 0
no_face_count = 0

print("\nEvaluating LFW images...")

for index, image_path in enumerate(unknown_images, start=1):

    vector = extract_embedding(image_path)

    if vector is None:
        no_face_count += 1
        continue

    # SVM prediction
    vector_2d = vector.reshape(1, -1)

    predicted_student = classifier.predict(vector_2d)[0]

    probabilities = classifier.predict_proba(vector_2d)[0]

    max_probability = float(np.max(probabilities))

    # Cosine similarity to each student centroid
    normalized_vector = normalize(vector)

    similarities = {}

    for student_id, centroid in student_centroids.items():

        similarity = float(
            np.dot(normalized_vector, centroid)
        )

        similarities[student_id] = similarity

    nearest_student = max(
        similarities,
        key=similarities.get
    )

    nearest_similarity = similarities[nearest_student]

    row = {
        "filename": image_path.name,
        "predicted_student": str(predicted_student),
        "svm_probability": max_probability,
        "nearest_student": nearest_student,
        "cosine_similarity": nearest_similarity
    }

    for threshold in COSINE_THRESHOLDS:
        row[f"accepted_at_{threshold:.2f}"] = (
            nearest_similarity >= threshold
        )

    results.append(row)

    processed_count += 1

    if index % 50 == 0:
        print(
            f"Processed {index}/{len(unknown_images)} images..."
        )


# ==================================================
# 9. SAVE CSV REPORT
# ==================================================

fieldnames = [
    "filename",
    "predicted_student",
    "svm_probability",
    "nearest_student",
    "cosine_similarity"
]

fieldnames.extend([
    f"accepted_at_{threshold:.2f}"
    for threshold in COSINE_THRESHOLDS
])

with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# ==================================================
# 10. PRINT RESULTS
# ==================================================

print("\n" + "=" * 60)
print("LFW UNKNOWN PERSON EVALUATION RESULTS")
print("=" * 60)

print("Total LFW images:", len(unknown_images))
print("Images successfully processed:", processed_count)
print("Images with no detected face:", no_face_count)

if processed_count > 0:

    print("\nExploratory cosine similarity results:")

    for threshold in COSINE_THRESHOLDS:

        accepted = sum(
            row[f"accepted_at_{threshold:.2f}"]
            for row in results
        )

        rejected = processed_count - accepted

        rejection_rate = (
            rejected / processed_count
        ) * 100

        print(
            f"\nThreshold: {threshold:.2f}"
            f"\nPotential false accepts: {accepted}"
            f"\nRejected unknown faces: {rejected}"
            f"\nUnknown rejection rate: {rejection_rate:.2f}%"
        )

    print("\nSVM predicted identity distribution:")

    for student_id in student_centroids:

        count = sum(
            row["predicted_student"] == student_id
            for row in results
        )

        print(f"{student_id}: {count}")

else:
    print("No LFW faces could be processed.")

print("\nCSV report:", CSV_PATH)

print("\nExisting student datasets were not modified.")
print("Existing trained model was not modified.")

print("\nEvaluation completed.")
