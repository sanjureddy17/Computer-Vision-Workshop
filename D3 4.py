import cv2

cap=cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    resized_frame = cv2.resize(frame, (700, 700))

    cv2.imshow("Video Manipulation", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()