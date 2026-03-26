
from ultralytics import YOLO
import cv2
import time

# ✅ Use better model
model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# ✅ Improve resolution
cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    print("Camera not detected")
    exit()

FILE_NAME = "attendance.txt"
last_count = -1

print("🔥 Auto saving when count changes | ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Use tracking
    results = model.track(frame, persist=True, classes=[0], conf=0.3)

    # ✅ Better counting
    count = 0
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            area = (x2 - x1) * (y2 - y1)

            if area > 5000:
                count += 1

    frame = results[0].plot()

    cv2.putText(frame, f"Humans: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Detection", frame)

    # ✅ Save only when count changes
    if count != last_count:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(FILE_NAME, "a") as f:
            f.write(f"{timestamp} - Count: {count}\n")

        print(f"[{timestamp}] Logged Count: {count}")

        last_count = count

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()