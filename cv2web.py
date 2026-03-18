from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# 🔥 AUTO-DETECT BEST CAMERA (iVCam)
def get_camera():
    best_index = 0
    best_width = 0

    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w, _ = frame.shape
                print(f"Camera {i}: {w}x{h}")

                # choose highest resolution (usually iVCam)
                if w > best_width:
                    best_width = w
                    best_index = i

            cap.release()

    print(f"Using camera index: {best_index}")
    return cv2.VideoCapture(best_index, cv2.CAP_DSHOW)


cap = get_camera()

if not cap.isOpened():
    print("❌ Camera not detected")
    exit()

# 🔁 Detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # Detect ONLY humans
    results = model(frame, classes=[0], conf=0.5)

    count = len(results[0].boxes)

    frame = results[0].plot()

    cv2.putText(frame, f"Humans: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("iVCam Human Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()