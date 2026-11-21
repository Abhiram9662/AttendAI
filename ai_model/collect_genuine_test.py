
from pathlib import Path
import cv2

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "dataset" / "genuine_test"

TARGET_IMAGES = 40
CAMERA_INDEX = 0

STUDENT_IDS = {"12407909", "12414409"}

# ==========================================
# 2. SELECT STUDENT
# ==========================================

student_id = input(
    "Enter student ID (12407909 or 12414409): "
).strip()

if student_id not in STUDENT_IDS:
    raise ValueError("Invalid student ID.")

student_dir = OUTPUT_DIR / student_id
student_dir.mkdir(parents=True, exist_ok=True)

# Do not overwrite existing test images.
existing = list(student_dir.glob("*.jpg"))

if existing:
    raise RuntimeError(
        f"{student_id} already has {len(existing)} images. "
        "No files were overwritten."
    )

# ==========================================
# 3. INITIALIZE FACE DETECTOR
# ==========================================

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

detector = cv2.CascadeClassifier(cascade_path)

if detector.empty():
    raise RuntimeError("Could not load face detector.")

# ==========================================
# 4. OPEN WEBCAM
# ==========================================

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check camera permissions."
    )

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

saved_count = 0

print("\nCamera started.")
print("Press S to save a photograph.")
print("Press Q to quit.")

# ==========================================
# 5. CAPTURE PHOTOGRAPHS
# ==========================================

try:

    while saved_count < TARGET_IMAGES:

        success, clean_frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        gray = cv2.cvtColor(
            clean_frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        display_frame = clean_frame.copy()

        # Draw rectangles around detected faces.
        for (x, y, w, h) in faces:

            cv2.rectangle(
                display_frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Display student information.
        cv2.putText(
            display_frame,
            f"Student: {student_id}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"Saved: {saved_count}/{TARGET_IMAGES}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            "S = Save | Q = Quit",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Genuine Student Test Capture",
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):

            if len(faces) != 1:
                print(
                    "Rejected: exactly one face must be visible."
                )
                continue

            filename = f"test_{saved_count + 1:03d}.jpg"

            output_path = student_dir / filename

            saved = cv2.imwrite(
                str(output_path),
                clean_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            if saved:
                saved_count += 1
                print(f"Saved: {filename}")
            else:
                print("Could not save image.")

finally:

    camera.release()
    cv2.destroyAllWindows()

# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print("\nCapture completed.")
print("Student:", student_id)
print("Images saved:", saved_count)
print("Output folder:", student_dir)
