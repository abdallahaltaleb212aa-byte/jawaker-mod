from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import uvicorn

app = FastAPI()

@app.post("/api/find_crop")
async def find_crop(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"found": False, "message": "Image error"}

    # بنجيب أبعاد الصورة اللي جاية من الموبايل
    img_height, img_width = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1500: # حجم القمح
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            # بنرجع الإحداثيات + أبعاد الصورة عشان الموبايل يحسبها صح
            return {"found": True, "x": center_x, "y": center_y, "width": img_width, "height": img_height}
            
    return {"found": False}

@app.get("/")
def read_root():
    return {"status": "HayDay Bot Server is Running!"}
