from pathlib import Path
from functools import lru_cache
from io import BytesIO
from datetime import datetime
import threading
import shutil
import uuid
import re
import json

import cv2
import joblib
import numpy as np
import torch

from PIL import Image, UnidentifiedImageError
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Depends,
    Form,
)
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.database import get_db
from app.models import User, Attendance, AttendanceSession
from facenet_pytorch import MTCNN, InceptionResnetV1


router = APIRouter(prefix="/api/vision", tags=["computer vision"])

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = (
    PROJECT_ROOT
    / "ai_model"
    / "trained_models"
    / "face_classifier.joblib"
)
RAW_DIR = PROJECT_ROOT / "ai_model" / "dataset" / "raw"
INACTIVE_STUDENTS_FILE = PROJECT_ROOT / "ai_model" / "dataset" / "inactive_students.json"

# Initial threshold selected from the user's evaluation.
# It is a prototype threshold, not a production-certified setting.
SIMILARITY_THRESHOLD = 0.70
MAX_IMAGE_SIZE = 5 * 1024 * 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Protect shared PyTorch model inference from overlapping requests.
INFERENCE_LOCK = threading.Lock()


def normalize(vector):
    """Return an L2-normalized vector, or None for a zero vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None
    return vector / norm




def model_student_ids_for_cache(classifier):
    """IDs represented by the unchanged, originally trained SVM classifier."""
    return {str(value) for value in getattr(classifier, "classes_", [])}


def _session_time_is_active(session: AttendanceSession) -> bool:
    """
    Compare the current local server time with the session interval.
    SQLite commonly returns naive datetimes; aware values are handled
    by using their timezone for the current-time comparison.
    """
    starts_at = session.starts_at
    ends_at = session.ends_at

    if starts_at.tzinfo is not None:
        now = datetime.now(starts_at.tzinfo)
    elif ends_at.tzinfo is not None:
        now = datetime.now(ends_at.tzinfo)
    else:
        now = datetime.now()

    # If one endpoint is aware and the other is naive, normalize the
    # naive value to the aware endpoint's timezone.
    if starts_at.tzinfo is None and ends_at.tzinfo is not None:
        starts_at = starts_at.replace(tzinfo=ends_at.tzinfo)
    elif starts_at.tzinfo is not None and ends_at.tzinfo is None:
        ends_at = ends_at.replace(tzinfo=starts_at.tzinfo)

    return starts_at <= now <= ends_at


@lru_cache(maxsize=1)
def load_recognition_models():
    """
    Load the existing SVM classifier, MTCNN and pretrained FaceNet.
    Build reference centroids from ai_model/dataset/raw.
    This function does not retrain or overwrite the classifier.
    """
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Trained classifier not found: {MODEL_PATH}")

    if not RAW_DIR.exists():
        raise RuntimeError(f"Student reference dataset not found: {RAW_DIR}")

    print("Loading existing face classifier...")
    model_data = joblib.load(MODEL_PATH)

    classifier = model_data["classifier"]
    model_student_ids = [str(student_id) for student_id in model_data["classes"]]
    # Include newly enrolled student folders without changing/retraining the
    # original SVM classifier. New students are verified by their centroid.
    folder_student_ids = sorted(
        path.name for path in RAW_DIR.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    inactive_ids = set()
    if INACTIVE_STUDENTS_FILE.exists():
        try:
            inactive_ids = {str(value) for value in json.loads(INACTIVE_STUDENTS_FILE.read_text(encoding="utf-8"))}
        except (OSError, ValueError, TypeError):
            raise RuntimeError("Inactive student registry is unreadable; refusing to load an unsafe recognition roster.")
    student_ids = sorted((set(model_student_ids) | set(folder_student_ids)) - inactive_ids)

    if model_data.get("embedding_dimension") != 512:
        raise RuntimeError("Expected 512-dimensional embeddings.")

    print("Registered students:", student_ids)
    print("Device:", DEVICE)

    # Match the preprocessing used by train.py/evaluate_genuine_test.py.
    reference_mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        post_process=True,
        device=DEVICE,
    )

    # keep_all=True allows multiple faces in one uploaded image.
    live_mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=True,
        post_process=True,
        device=DEVICE,
    )

    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

    centroids = {}
    print("Building reference face centroids...")

    for student_id in student_ids:
        student_folder = RAW_DIR / student_id
        if not student_folder.exists():
            if student_id in model_student_ids:
                raise RuntimeError(f"Missing reference folder: {student_folder}")
            continue

        image_files = sorted(
            [
                *student_folder.glob("*.jpg"),
                *student_folder.glob("*.jpeg"),
                *student_folder.glob("*.png"),
            ]
        )

        vectors = []
        for image_path in image_files:
            try:
                image = Image.open(image_path).convert("RGB")
                face = reference_mtcnn(image)
                if face is None:
                    continue

                face = face.unsqueeze(0).to(DEVICE)
                with torch.inference_mode():
                    embedding = facenet(face)

                vector = embedding.cpu().numpy()[0]
                vector = normalize(vector)
                if vector is not None:
                    vectors.append(vector)

            except Exception as error:
                print(f"Reference image error: {image_path.name}: {error}")

        if not vectors:
            raise RuntimeError(
                f"No valid reference embeddings for {student_id}"
            )

        centroid = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
        centroid = normalize(centroid)
        if centroid is None:
            raise RuntimeError(f"Invalid reference centroid for {student_id}")

        centroids[student_id] = centroid
        print(f"{student_id}: {len(vectors)} reference embeddings")

    print("Face recognition models loaded successfully.")
    return classifier, live_mtcnn, facenet, centroids


# ---------------------------------------------------------------------
# EXISTING FACE DETECTION ENDPOINT — PRESERVED
# ---------------------------------------------------------------------

@router.post("/detect")
async def detect_faces(image: UploadFile = File(...)):
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image",
        )

    raw = await image.read()
    if len(raw) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be 5 MB or smaller",
        )

    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = (
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    return {
        "face_count": len(faces),
        "faces": [
            {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
            }
            for x, y, w, h in faces
        ],
        "image_width": int(frame.shape[1]),
        "image_height": int(frame.shape[0]),
        "notice": (
            "Detection only. No identity recognition or attendance "
            "was performed."
        ),
    }


# ---------------------------------------------------------------------
# ACTIVE STUDENT ROSTER + SAFE STUDENT ARCHIVING
# ---------------------------------------------------------------------

@router.get("/students")
def list_active_students(db: Session = Depends(get_db)):
    """Return only active students for the dashboard roster."""
    users = (
        db.query(User)
        .filter(User.role != "inactive")
        .order_by(User.student_id.asc())
        .all()
    )
    return [
        {
            "id": user.id,
            "student_id": user.student_id,
            "full_name": user.full_name,
            "email": user.email,
            "department": user.department,
            "section": user.section,
        }
        for user in users
    ]


@router.delete("/students/{student_id}")
def delete_student(student_id: str, db: Session = Depends(get_db)):
    """Permanently remove the student record and attendance rows.

    For IDs in the original frozen classifier, retain the original training
    folder as a backup and add the ID to the inactive registry so recognition
    rejects it. Newly enrolled photo folders are removed.
    """
    student = db.query(User).filter(User.student_id == student_id).first()
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found.")

    inactive_ids = set()
    if INACTIVE_STUDENTS_FILE.exists():
        try:
            inactive_ids = {
                str(value)
                for value in json.loads(INACTIVE_STUDENTS_FILE.read_text(encoding="utf-8"))
            }
        except (OSError, ValueError, TypeError):
            raise HTTPException(
                status_code=500,
                detail="Inactive student registry is unreadable; no changes made.",
            )

    original_model_ids = set()
    if MODEL_PATH.exists():
        try:
            model_data = joblib.load(MODEL_PATH)
            original_model_ids = {str(value) for value in model_data.get("classes", [])}
        except Exception as error:
            print("Could not inspect original classifier classes:", repr(error))
            raise HTTPException(
                status_code=500,
                detail="Could not safely inspect the original model; no changes made.",
            )

    # Prepare the inactive registry first so a student from the frozen SVM
    # cannot be recognized after the database record is removed.
    inactive_ids.add(student_id)
    INACTIVE_STUDENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_registry = INACTIVE_STUDENTS_FILE.with_suffix(".tmp")
    temp_registry.write_text(json.dumps(sorted(inactive_ids), indent=2), encoding="utf-8")

    try:
        temp_registry.replace(INACTIVE_STUDENTS_FILE)

        # This is a permanent deletion: remove attendance rows to satisfy the
        # existing foreign key, then remove the student database row.
        deleted_attendance = db.query(Attendance).filter(
            Attendance.user_id == student.id
        ).delete(synchronize_session=False)

        db.delete(student)
        db.commit()
    except Exception as error:
        db.rollback()
        temp_registry.unlink(missing_ok=True)
        print("Student deletion error:", repr(error))
        raise HTTPException(
            status_code=500,
            detail="Could not delete student from the database. Check backend logs.",
        )

    face_folder = RAW_DIR / student_id
    photos_removed = False
    original_training_data_preserved = student_id in original_model_ids
    if not original_training_data_preserved and face_folder.exists():
        try:
            shutil.rmtree(face_folder)
            photos_removed = True
        except OSError as error:
            print("Could not remove enrollment photo folder:", repr(error))

    load_recognition_models.cache_clear()
    return {
        "message": "Student permanently deleted from the backend.",
        "student_id": student_id,
        "database_record_deleted": True,
        "attendance_records_deleted": int(deleted_attendance or 0),
        "enrollment_photos_removed": photos_removed,
        "original_training_data_preserved": original_training_data_preserved,
        "recognition_blocked": True,
    }


# ---------------------------------------------------------------------
# NEW STUDENT ENROLLMENT — 30 WEBCAM PHOTOS
# ---------------------------------------------------------------------

ENROLLMENT_PHOTO_COUNT = 30
MAX_ENROLLMENT_IMAGE_SIZE = 2 * 1024 * 1024


@router.post("/enroll")
async def enroll_student(
    student_id: str = Form(...),
    full_name: str = Form(...),
    email: str = Form(...),
    department: str = Form(""),
    section: str = Form(""),
    images: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """Create a student and store exactly 30 uploaded image files.

    Enrollment deliberately does NOT run face detection or face-count checks.
    The files are stored as reference images; images without usable faces may
    later be skipped when recognition centroids are built.
    Existing users, original raw images and trained SVM file are not overwritten.
    """
    student_id = student_id.strip()
    full_name = full_name.strip()
    email = email.strip().lower()
    department = department.strip()
    section = section.strip()

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", student_id):
        raise HTTPException(
            status_code=422,
            detail="Student ID may contain only letters, numbers, hyphens and underscores (maximum 50 characters).",
        )
    if len(full_name) < 2 or len(full_name) > 150:
        raise HTTPException(status_code=422, detail="Enter a valid full name.")
    if len(email) < 5 or len(email) > 255 or "@" not in email:
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    if len(images) != ENROLLMENT_PHOTO_COUNT:
        raise HTTPException(
            status_code=422,
            detail=f"Exactly {ENROLLMENT_PHOTO_COUNT} image files are required; received {len(images)}.",
        )

    if db.query(User).filter(User.student_id == student_id).first():
        raise HTTPException(status_code=409, detail="This student ID is already registered.")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="This email address is already registered.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    final_dir = RAW_DIR / student_id
    if final_dir.exists():
        raise HTTPException(
            status_code=409,
            detail="A face dataset folder already exists for this student ID. Nothing was overwritten.",
        )

    staging_dir = RAW_DIR / f".enroll_{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    saved_count = 0

    try:
        # Upload-only mode: validate file format/readability/size, but do not
        # invoke MTCNN or reject images based on whether a face is detected.
        for index, upload in enumerate(images, start=1):
            if upload.content_type not in {"image/jpeg", "image/png", "image/webp"}:
                raise HTTPException(status_code=415, detail=f"Image {index}: use JPEG, PNG or WebP.")

            raw = await upload.read()
            if not raw:
                raise HTTPException(status_code=400, detail=f"Image {index} is empty.")
            if len(raw) > MAX_ENROLLMENT_IMAGE_SIZE:
                raise HTTPException(status_code=413, detail=f"Image {index} exceeds 2 MB.")

            try:
                image = Image.open(BytesIO(raw)).convert("RGB")
                image.verify if False else None
            except (UnidentifiedImageError, OSError):
                raise HTTPException(status_code=400, detail=f"Image {index} is not a readable image.")

            # Re-open after conversion and save a consistent JPEG copy.
            try:
                image = Image.open(BytesIO(raw)).convert("RGB")
            except (UnidentifiedImageError, OSError):
                raise HTTPException(status_code=400, detail=f"Image {index} is not a readable image.")

            output_path = staging_dir / f"{index:03d}.jpg"
            image.save(output_path, format="JPEG", quality=95, optimize=True)
            saved_count += 1

        user = User(
            student_id=student_id,
            full_name=full_name,
            email=email,
            department=department,
            section=section,
        )
        db.add(user)
        db.flush()

        staging_dir.rename(final_dir)
        try:
            db.commit()
            db.refresh(user)
        except Exception:
            db.rollback()
            shutil.rmtree(final_dir, ignore_errors=True)
            raise

        # Clear only the in-memory centroid cache. No model file is retrained
        # or overwritten; unusable/no-face images are skipped during embedding.
        load_recognition_models.cache_clear()

        return {
            "message": "Student registered and 30 image files uploaded. Face detection was not performed during enrollment.",
            "student": {
                "id": user.id,
                "student_id": user.student_id,
                "full_name": user.full_name,
                "email": user.email,
                "department": user.department,
                "section": user.section,
            },
            "photos_saved": saved_count,
            "face_validation": "not_performed",
            "model_updated": "centroid cache refreshed on next recognition; original SVM classifier preserved",
        }

    except HTTPException:
        db.rollback()
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    except Exception as error:
        db.rollback()
        shutil.rmtree(staging_dir, ignore_errors=True)
        print("Student enrollment error:", repr(error))
        raise HTTPException(status_code=500, detail="Student enrollment failed. Check backend terminal logs.")
    finally:
        for upload in images:
            await upload.close()


# ---------------------------------------------------------------------
# FACE RECOGNITION + AUTOMATIC ATTENDANCE ENDPOINT
# ---------------------------------------------------------------------

@router.post("/recognize")
async def recognize_faces(
    image: UploadFile = File(...),
    session_id: int = Form(...),
    db: Session = Depends(get_db),
):
    # Validate the image upload.
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image",
        )

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    if len(raw) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be 5 MB or smaller",
        )

    try:
        pil_image = Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Check session before running the expensive face-recognition model.
    attendance_session = (
        db.query(AttendanceSession)
        .filter(AttendanceSession.id == session_id)
        .first()
    )

    if attendance_session is None:
        raise HTTPException(status_code=404, detail="Attendance session not found")

    if attendance_session.status != "open":
        raise HTTPException(
            status_code=400,
            detail="Attendance session is not open",
        )

    if not _session_time_is_active(attendance_session):
        raise HTTPException(
            status_code=400,
            detail="Attendance session is outside its scheduled time",
        )

    try:
        with INFERENCE_LOCK:
            classifier, mtcnn, facenet, centroids = load_recognition_models()

            boxes, probabilities = mtcnn.detect(pil_image)

            if boxes is None:
                return {
                    "session_id": session_id,
                    "face_count": 0,
                    "recognized_count": 0,
                    "unknown_count": 0,
                    "threshold": SIMILARITY_THRESHOLD,
                    "faces": [],
                    "attendance": [],
                    "message": "No face detected. No attendance was recorded.",
                }

            face_crops = mtcnn.extract(
                pil_image,
                boxes,
                save_path=None,
            )

            if face_crops is None:
                return {
                    "session_id": session_id,
                    "face_count": 0,
                    "recognized_count": 0,
                    "unknown_count": 0,
                    "threshold": SIMILARITY_THRESHOLD,
                    "faces": [],
                    "attendance": [],
                    "message": "No usable face detected. No attendance was recorded.",
                }

            if face_crops.ndim == 3:
                face_crops = face_crops.unsqueeze(0)

            face_crops = face_crops.to(DEVICE)

            with torch.inference_mode():
                embeddings = facenet(face_crops)

            vectors = embeddings.cpu().numpy().astype(np.float32)

            if vectors.ndim != 2 or vectors.shape[1] != 512:
                raise RuntimeError("FaceNet returned an unexpected embedding shape")

            svm_predictions = classifier.predict(vectors)

            normalized_vectors = []
            for vector in vectors:
                normalized = normalize(vector)
                if normalized is None:
                    normalized = np.zeros_like(vector)
                normalized_vectors.append(normalized)

            normalized_vectors = np.asarray(
                normalized_vectors,
                dtype=np.float32,
            )

            student_ids = list(centroids.keys())
            centroid_matrix = np.asarray(
                [centroids[student_id] for student_id in student_ids],
                dtype=np.float32,
            )

            similarities = np.dot(normalized_vectors, centroid_matrix.T)

            results = []
            for index, box in enumerate(boxes):
                if box is None:
                    continue

                svm_id = str(svm_predictions[index])
                nearest_index = int(np.argmax(similarities[index]))
                nearest_id = student_ids[nearest_index]
                similarity = float(similarities[index][nearest_index])

                # Require both a passing similarity and agreement between
                # the SVM prediction and the nearest reference centroid.
                # Existing trained classes still require SVM/centroid agreement.
                # Newly enrolled IDs are not in the frozen SVM classes; they use
                # their freshly built centroid plus the same similarity threshold.
                is_original_model_class = nearest_id in model_student_ids_for_cache(classifier)
                recognized = (
                    similarity >= SIMILARITY_THRESHOLD
                    and (svm_id == nearest_id or not is_original_model_class)
                )

                x1, y1, x2, y2 = box.tolist()

                detection_probability = None
                if probabilities is not None and probabilities[index] is not None:
                    detection_probability = round(
                        float(probabilities[index]), 4
                    )

                results.append(
                    {
                        "face_index": index + 1,
                        "student_id": nearest_id if recognized else None,
                        "status": "recognized" if recognized else "unknown",
                        "similarity": round(similarity, 4),
                        "svm_prediction": svm_id,
                        "nearest_centroid": nearest_id,
                        "svm_centroid_agreement": svm_id == nearest_id,
                        "detection_probability": detection_probability,
                        "bounding_box": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                        },
                    }
                )

        # Attendance database operations happen after inference, outside
        # the model lock. Unknown faces are never sent to the database.
        attendance_results = []
        processed_students = set()

        for item in results:
            if item["status"] != "recognized":
                continue

            student_id = item["student_id"]

            # Avoid marking the same student twice from one image.
            if student_id in processed_students:
                attendance_results.append(
                    {
                        "student_id": student_id,
                        "status": "duplicate_face_in_image",
                    }
                )
                continue

            processed_students.add(student_id)

            user = (
                db.query(User)
                .filter(User.student_id == student_id)
                .first()
            )

            if user is None:
                attendance_results.append(
                    {
                        "student_id": student_id,
                        "status": "student_not_registered",
                    }
                )
                continue

            existing = (
                db.query(Attendance)
                .filter(
                    Attendance.user_id == user.id,
                    Attendance.session_id == session_id,
                )
                .first()
            )

            if existing is not None:
                attendance_results.append(
                    {
                        "student_id": student_id,
                        "status": "already_marked",
                        "attendance_id": existing.id,
                    }
                )
                continue

            attendance = Attendance(
                user_id=user.id,
                session_id=session_id,
                status="present",
                method="face_recognition",
            )
            db.add(attendance)

            try:
                db.commit()
                db.refresh(attendance)
                attendance_results.append(
                    {
                        "student_id": student_id,
                        "status": "recorded",
                        "attendance_id": attendance.id,
                    }
                )

            except IntegrityError:
                db.rollback()

                # The unique constraint is the final protection against
                # duplicate attendance if two requests arrive together.
                existing = (
                    db.query(Attendance)
                    .filter(
                        Attendance.user_id == user.id,
                        Attendance.session_id == session_id,
                    )
                    .first()
                )

                if existing is not None:
                    attendance_results.append(
                        {
                            "student_id": student_id,
                            "status": "already_marked",
                            "attendance_id": existing.id,
                        }
                    )
                else:
                    raise HTTPException(
                        status_code=409,
                        detail="Could not save attendance",
                    )

        recognized_count = sum(
            item["status"] == "recognized" for item in results
        )
        unknown_count = sum(
            item["status"] == "unknown" for item in results
        )

        return {
            "session_id": session_id,
            "face_count": len(results),
            "recognized_count": recognized_count,
            "unknown_count": unknown_count,
            "threshold": SIMILARITY_THRESHOLD,
            "faces": results,
            "attendance": attendance_results,
            "notice": (
                "Prototype face recognition and attendance recording. "
                "Liveness verification is not implemented; an uploaded "
                "photograph alone does not prove that a live person is present."
            ),
        }

    except HTTPException:
        raise

    except Exception as error:
        db.rollback()
        print("Recognition error:", repr(error))
        raise HTTPException(
            status_code=500,
            detail="Face recognition failed. Check backend terminal logs.",
        )
