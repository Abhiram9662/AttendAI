from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np

router = APIRouter(prefix="/api/vision", tags=["computer vision"])

@router.post("/detect")
async def detect_faces(image: UploadFile = File(...)):
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")
    raw = await image.read()
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 5 MB or smaller")
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return {
        "face_count": len(faces),
        "faces": [{"x": int(x), "y": int(y), "width": int(w), "height": int(h)} for x, y, w, h in faces],
        "image_width": int(frame.shape[1]),
        "image_height": int(frame.shape[0]),
        "notice": "Detection only. No identity recognition or attendance was performed."
    }
