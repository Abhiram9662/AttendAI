
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "raw"

DATASET_DIR.mkdir(parents=True, exist_ok=True)

student_id = input("Enter Student ID: ").strip()

if not student_id or any(
    c in student_id for c in r'\/:*?"<>|'
):
    raise ValueError("Invalid student ID")

student_dir = DATASET_DIR / student_id
student_dir.mkdir(parents=True, exist_ok=True)

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Cannot access webcam")

saved = 0
MAX_IMAGES = 50
frame_count = 0

print("Camera started.")
print("Look at camera and change your head angle.")
print("Press Q to stop.")

try:
    while saved < MAX_IMAGES:

        ret, frame = camera.read()

        if not ret:
            break

        frame_count += 1

        gray = cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        )

        faces = detector.detectMultiScale(
            gray,
            1.1,
            5,
            minSize=(100, 100)
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            if len(faces) == 1 and frame_count % 8 == 0:

                face = frame[y:y+h, x:x+w]

                filename = (
                    student_dir /
                    f"{student_id}_{saved+1:04d}.jpg"
                )

                cv2.imwrite(str(filename), face)
                saved += 1

        cv2.putText(
            frame,
            f"Images: {saved}/{MAX_IMAGES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Face Dataset Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"Saved {saved} images.")
print("Student folder:", student_dir)
