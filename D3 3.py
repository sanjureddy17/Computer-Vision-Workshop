import cv2

cap=cv2.VideoCapture('CarVideo.mp4')

while True:
    ret,frame=cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.imshow("Grayscale video",gray)

    if cv2.waitKey(25) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()