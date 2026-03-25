from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # your iVCam index

if not cap.isOpened():
    print("Camera not detected")
    exit()

FILE_NAME = "attendance.txt"

last_count = -1  # track previous count

print("🔥 Auto saving when count changes | ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    results = model(frame, classes=[0], conf=0.5)
    count = len(results[0].boxes)

    frame = results[0].plot()

    cv2.putText(frame, f"Humans: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Detection", frame)

    # 🔥 SAVE ONLY WHEN COUNT CHANGES
    if count != last_count:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(FILE_NAME, "a") as f:
            f.write(f"{timestamp} - Count: {count}\n")

        print(f"[{timestamp}] Logged Count: {count}")

        last_count = count  # update

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()