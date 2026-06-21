
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import numpy as np
import cv2
import os
import time
from datetime import datetime
from fastapi.responses import FileResponse
from fastapi import HTTPException

app = FastAPI()

model = YOLO('best1.pt')

os.makedirs("detections", exist_ok=True)

@app.get("/")
def home():
    return {"status": "API is running"}

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

@app.get("/image/{filename}")
def get_image(filename: str):

    image_path = os.path.join("detections", filename)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # start time
    start_time = time.time()

    # read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # inference
    try:
        results = model.predict(
            img,
            conf=0.35,
            verbose=False
        )

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

    detections = []
    fire_detected = False
    smoke_detected = False

    for box in results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.4:
            continue

        class_name = model.names[cls]
        if class_name in ["light", "nonfire"]:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class": class_name,
            "confidence": round(conf, 2),
            "bbox": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            }
        })

        if class_name == "fire":
            fire_detected = True

        if class_name == "smoke":
            smoke_detected = True

    # save image if fire
    saved_image = None

    if fire_detected or smoke_detected:

        # copy image
        saved_frame = img.copy()

        # رسم الـ boxes
        for det in detections:

            x1 = det["bbox"]["x1"]
            y1 = det["bbox"]["y1"]
            x2 = det["bbox"]["x2"]
            y2 = det["bbox"]["y2"]

            label = f"{det['class']} {det['confidence']}"

                # ألوان
            color = (0,255,0)

            if det["class"] == "fire":
                color = (0,0,255)

            elif det["class"] == "smoke":
                color = (255,255,0)

            # rectangle
            cv2.rectangle(
                saved_frame,
                (x1,y1),
                (x2,y2),
                color,
                2
            )

            # text
            cv2.putText(
                saved_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # اسم الصورة
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        detection_type = "fire"

        if smoke_detected and not fire_detected:
            detection_type = "smoke"

        saved_image = f"detections/{detection_type}_{timestamp}.jpg"

        # حفظ الصورة
        cv2.imwrite(saved_image, saved_frame)

    # processing time
    processing_time = round((time.time() - start_time) * 1000, 2)

    # message
    if fire_detected:
        message = "Fire detected"
    elif smoke_detected:
        message = "Smoke detected"
    else:
        message = "No fire or smoke detected"

    BASE_URL = "http://127.0.0.1:8000"  # أو رابط السيرفر بعد النشر

    image_url = None

    if saved_image:
        image_url = f"{BASE_URL}/image/{os.path.basename(saved_image)}"

    return {
    "success": True,
    "message": message,
    "fire_detected": fire_detected,
    "smoke_detected": smoke_detected,
    "total_detections": len(detections),
    "processing_time_ms": processing_time,
    "timestamp": datetime.now().isoformat(),
    "saved_image": saved_image,
    "image_url": image_url,
    "detections": detections
    }


