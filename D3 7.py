import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    inverted = cv2.bitwise_not(frame)

    cv2.imshow("Original", frame)
    cv2.imshow("Inverted Video", inverted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()