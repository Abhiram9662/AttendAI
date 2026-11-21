
from pathlib import Path
import numpy as np
import torch
import joblib

from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset" / "raw"
MODEL_DIR = BASE_DIR / "trained_models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "face_classifier.joblib"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 50)
print("FACE RECOGNITION MODEL TRAINING")
print("=" * 50)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 2. INITIALIZE FACE DETECTION
# ==========================================

print("\nLoading MTCNN...")

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    post_process=True,
    device=DEVICE
)


# ==========================================
# 3. LOAD PRETRAINED FACENET
# ==========================================

print("Loading pretrained FaceNet...")

facenet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(DEVICE)


# ==========================================
# 4. READ STUDENT DATASET
# ==========================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_DIR}"
    )

student_folders = sorted([
    folder for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
])

if len(student_folders) < 2:
    raise ValueError(
        "At least two student folders are required."
    )

embeddings = []
labels = []

print("\nStudents detected:", len(student_folders))


# ==========================================
# 5. EXTRACT FACENET EMBEDDINGS
# ==========================================

for student_folder in student_folders:

    student_id = student_folder.name

    image_files = sorted([
        *student_folder.glob("*.jpg"),
        *student_folder.glob("*.jpeg"),
        *student_folder.glob("*.png")
    ])

    print(f"\nProcessing student: {student_id}")
    print("Available images:", len(image_files))

    valid_count = 0

    for image_path in image_files:

        try:
            image = Image.open(image_path).convert("RGB")

            # Detect and align face.
            face = mtcnn(image)

            if face is None:
                print("Face not detected:", image_path.name)
                continue

            face = face.unsqueeze(0).to(DEVICE)

            # Generate 512-dimensional embedding.
            with torch.inference_mode():

                embedding = facenet(face)

            vector = embedding.cpu().numpy()[0]

            embeddings.append(vector)
            labels.append(student_id)

            valid_count += 1

        except Exception as error:

            print("Image error:", image_path.name)
            print(error)

    print("Valid face embeddings:", valid_count)


# ==========================================
# 6. PREPARE TRAINING DATA
# ==========================================

X = np.asarray(embeddings, dtype=np.float32)
y = np.asarray(labels)

if len(X) == 0:
    raise RuntimeError(
        "No valid face embeddings were generated."
    )

classes, counts = np.unique(
    y, return_counts=True
)

print("\nTotal valid images:", len(X))
print("Identities:", classes)

for student_id, count in zip(classes, counts):
    print(f"{student_id}: {count} images")

if len(classes) < 2 or np.min(counts) < 2:
    raise RuntimeError(
        "Need at least two identities and two valid images per identity."
    )


# ==========================================
# 7. SPLIT TRAINING AND TEST DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))


# ==========================================
# 8. TRAIN SVM CLASSIFIER
# ==========================================

print("\nTraining SVM classifier...")

classifier = make_pipeline(

    StandardScaler(),

    SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced",
        random_state=42
    )
)

classifier.fit(X_train, y_train)


# ==========================================
# 9. EVALUATE MODEL
# ==========================================

predictions = classifier.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 50)
print("TRAINING RESULTS")
print("=" * 50)

print(
    f"Validation accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==========================================
# 10. SAVE TRAINED CLASSIFIER
# ==========================================

model_data = {

    "classifier": classifier,

    "classes": classes,

    "embedding_dimension": 512,

    "model_name": "FaceNet VGGFace2 + SVM",

    "student_count": len(classes),

    "device_used": str(DEVICE)

}

joblib.dump(
    model_data,
    MODEL_PATH
)

print("\n" + "=" * 50)
print("MODEL TRAINING COMPLETED")
print("=" * 50)

print("Saved model:", MODEL_PATH)

print("GPU used:", DEVICE)

print("Registered students:", classes)

print("\nYour recognition classifier is ready!")
