import cv2

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Webcam-Press 's' to save,'q' to quit",frame)

    key=cv2.waitKey(1)

    if key == ord('s'):
        cv2.imwrite("captured_image.jpg",frame)
        print("Image saved successfully!")

    elif key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()