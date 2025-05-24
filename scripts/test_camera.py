import cv2

# On macOS you can try the AVFoundation backend:
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Failed to open camera")
    exit()

print("Camera opened. Streaming… (press q to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed, stopping.")
        break

    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
