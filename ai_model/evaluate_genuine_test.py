
from pathlib import Path
import csv

import joblib
import numpy as np
import torch

from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "dataset" / "raw"
TEST_DIR = BASE_DIR / "dataset" / "genuine_test"

MODEL_PATH = BASE_DIR / "trained_models" / "face_classifier.joblib"

LFW_CSV = BASE_DIR / "evaluation_results" / "lfw_unknown_results.csv"

RESULT_DIR = BASE_DIR / "evaluation_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULT_DIR / "genuine_test_results.csv"

STUDENT_IDS = ["12407909", "12414409"]

THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# 2. CHECK FILES
# ==================================================

for path in [RAW_DIR, TEST_DIR, MODEL_PATH, LFW_CSV]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file/folder: {path}")

for student_id in STUDENT_IDS:
    folder = TEST_DIR / student_id

    if not folder.exists():
        raise FileNotFoundError(f"Missing test folder: {folder}")

    images = list(folder.glob("*.jpg"))

    if len(images) < 1:
        raise RuntimeError(f"No test images found for {student_id}")


# ==================================================
# 3. LOAD EXISTING CLASSIFIER
# ==================================================

print("=" * 60)
print("GENUINE STUDENT RECOGNITION EVALUATION")
print("=" * 60)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

model_data = joblib.load(MODEL_PATH)
classifier = model_data["classifier"]

print("Loaded existing classifier.")
print("Registered students:", model_data["classes"])


# ==================================================
# 4. LOAD FACENET
# ==================================================

print("\nLoading MTCNN and FaceNet...")

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    post_process=True,
    device=DEVICE
)

facenet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(DEVICE)


# ==================================================
# 5. EMBEDDING FUNCTIONS
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

        return embedding.cpu().numpy()[0].astype(np.float32)

    except Exception as error:
        print(f"Error: {image_path.name}: {error}")
        return None


def normalize(vector):

    norm = np.linalg.norm(vector)

    if norm == 0:
        return None

    return vector / norm


def get_images(folder):

    return sorted([
        *folder.glob("*.jpg"),
        *folder.glob("*.jpeg"),
        *folder.glob("*.png")
    ])


# ==================================================
# 6. BUILD REFERENCE EMBEDDINGS
# ==================================================

print("\nBuilding student reference embeddings...")

centroids = {}

for student_id in STUDENT_IDS:

    folder = RAW_DIR / student_id
    vectors = []

    for image_path in get_images(folder):

        vector = extract_embedding(image_path)

        if vector is not None:
            vector = normalize(vector)

            if vector is not None:
                vectors.append(vector)

    if not vectors:
        raise RuntimeError(
            f"No valid reference embeddings for {student_id}"
        )

    centroid = np.mean(np.asarray(vectors), axis=0)
    centroid = normalize(centroid)

    centroids[student_id] = centroid

    print(
        f"{student_id}: {len(vectors)} reference embeddings"
    )


# ==================================================
# 7. EVALUATE GENUINE TEST IMAGES
# ==================================================

print("\nEvaluating new student photographs...")

results = []
no_face = 0

for true_student_id in STUDENT_IDS:

    folder = TEST_DIR / true_student_id
    image_files = get_images(folder)

    print(f"\nTesting student: {true_student_id}")
    print("Images:", len(image_files))

    for image_path in image_files:

        vector = extract_embedding(image_path)

        if vector is None:
            no_face += 1
            continue

        vector_2d = vector.reshape(1, -1)

        predicted_id = str(classifier.predict(vector_2d)[0])

        probabilities = classifier.predict_proba(vector_2d)[0]
        svm_probability = float(np.max(probabilities))

        normalized_vector = normalize(vector)

        similarities = {}

        for student_id, centroid in centroids.items():

            similarities[student_id] = float(
                np.dot(normalized_vector, centroid)
            )

        nearest_id = max(
            similarities,
            key=similarities.get
        )

        nearest_similarity = similarities[nearest_id]

        correct_svm = predicted_id == true_student_id

        correct_nearest = nearest_id == true_student_id

        row = {
            "filename": image_path.name,
            "true_student": true_student_id,
            "predicted_student": predicted_id,
            "svm_probability": svm_probability,
            "nearest_student": nearest_id,
            "cosine_similarity": nearest_similarity,
            "correct_svm": correct_svm,
            "correct_nearest": correct_nearest
        }

        for threshold in THRESHOLDS:

            accepted = nearest_similarity >= threshold

            row[f"accepted_{threshold:.2f}"] = accepted

            row[f"correctly_accepted_{threshold:.2f}"] = (
                accepted and correct_svm
            )

        results.append(row)


# ==================================================
# 8. SAVE GENUINE RESULTS
# ==================================================

fieldnames = list(results[0].keys()) if results else []

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:

    if fieldnames:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)


# ==================================================
# 9. LOAD UNKNOWN-PERSON RESULTS
# ==================================================

with open(LFW_CSV, newline="", encoding="utf-8") as file:
    lfw_results = list(csv.DictReader(file))

print("\nLFW unknown images:", len(lfw_results))


# ==================================================
# 10. CALCULATE METRICS
# ==================================================

total_genuine = len(results)

if total_genuine == 0:
    raise RuntimeError("No genuine test images were successfully processed.")

correct_svm_count = sum(
    row["correct_svm"] for row in results
)

correct_nearest_count = sum(
    row["correct_nearest"] for row in results
)

print("\n" + "=" * 60)
print("GENUINE RECOGNITION RESULTS")
print("=" * 60)

print("Genuine images processed:", total_genuine)
print("Images without detected faces:", no_face)

print(
    f"SVM identification accuracy: "
    f"{correct_svm_count / total_genuine * 100:.2f}%"
)

print(
    f"Nearest-centroid identification accuracy: "
    f"{correct_nearest_count / total_genuine * 100:.2f}%"
)


# ==================================================
# 11. FAR AND FRR AT EACH THRESHOLD
# ==================================================

print("\n" + "=" * 60)
print("COMBINED GENUINE AND UNKNOWN EVALUATION")
print("=" * 60)

for threshold in THRESHOLDS:

    key = f"accepted_at_{threshold:.2f}"

    # Genuine acceptance requires BOTH:
    # similarity threshold passed AND correct SVM identity.
    genuine_accepted = sum(
        row[f"correctly_accepted_{threshold:.2f}"]
        for row in results
    )

    genuine_rejected = total_genuine - genuine_accepted

    # LFW false accepts: unknown images passing similarity threshold.
    unknown_accepted = sum(
        str(row[key]).strip().lower() in {"true", "1"}
        for row in lfw_results
    )

    unknown_total = len(lfw_results)

    frr = genuine_rejected / total_genuine * 100

    far = (
        unknown_accepted / unknown_total * 100
        if unknown_total else 0
    )

    genuine_acceptance = (
        genuine_accepted / total_genuine * 100
    )

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Genuine acceptance rate: {genuine_acceptance:.2f}%")
    print(f"FRR: {frr:.2f}%")
    print(f"Unknown false accepts: {unknown_accepted}")
    print(f"FAR: {far:.2f}%")


# ==================================================
# 12. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print("Genuine report:", OUTPUT_CSV)
print("LFW report:", LFW_CSV)

print("Existing model was not modified.")
print("Original student images were not modified.")
print("Genuine test images were not modified.")
