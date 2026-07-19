import cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("Img2.jpg")

cv2.rectangle(
    img,
    (50,50),
    (300,200),
    (0,255,0),
    3
)

cv2.circle(
    img,
    (500,150),
    80,
    (255,0,0),
    3
)

cv2.putText(
    img,
    "Welcome to Innomatics",
    (160,300),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0,0,255),
    2,
    cv2.LINE_AA
)

plt.figure(figsize=(10,6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

