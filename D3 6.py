import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create black mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Draw white circle mask
    cv2.circle(mask, (frame.shape[1]//2, frame.shape[0]//2), 150, 255, -1)

    # Apply mask
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Original", frame)
    cv2.imshow("Masked Video", masked_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()