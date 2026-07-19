import cv2
Import the OpenCV library for computer vision operations
# Open the default webcam (0 = primary webcam)
cap = cv2.VideoCapture(0)
import cv2

# Import the time module for FPS (Frames Per Second) calculation
import time

# --------------------------------------------------
# Load Haar Cascade
# --------------------------------------------------

# Load the pre-trained Haar Cascade classifier for frontal face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Check whether the Haar Cascade file was loaded successfully
if face_cascade.empty():
    print("Error loading Haar Cascade.")
    exit()

# --------------------------------------------------
# Open Webcam
# --------------------------------------------------

# Open the default webcam (0 = primary webcam)
cap = cv2.VideoCapture(0)

# Check whether the webcam opened successfully
if not cap.isOpened():
    print("Unable to open webcam.")
    exit()

# Set webcam resolution to 1280 × 720 (HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Store the initial time for FPS calculation
prev_time = time.time()

# --------------------------------------------------
# Main Loop
# --------------------------------------------------

# Start an infinite loop to continuously process webcam frames
while True:

    # Capture one frame from the webcam
    # ret   -> True if frame is captured successfully
    # frame -> Captured image
    ret, frame = cap.read()

    # Exit the loop if frame capture fails
    if not ret:
        break

    # Convert the color image into grayscale
    # Haar Cascade works faster on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(

        # Input grayscale image
        gray,

        # Image scaling factor for multi-scale detection
        scaleFactor=1.1,

        # Number of neighboring detections required
        # Higher value = fewer false detections
        minNeighbors=5,

        # Minimum face size to detect
        minSize=(40, 40)
    )

    # Draw Bounding Boxes
    # Loop through every detected face
    for (x, y, w, h) in faces:

        # Draw a green rectangle around the detected face
        cv2.rectangle(

            # Input image
            frame,

            # Top-left corner
            (x, y),

            # Bottom-right corner
            (x + w, y + h),

            # Rectangle color (Green)
            (0, 255, 0),

            # Rectangle thickness
            2
        )

        # Display the label "Face" above the bounding box
        cv2.putText(

            # Input image
            frame,

            # Text to display
            "Face",

            # Text position
            (x, y - 10),

            # Font style
            cv2.FONT_HERSHEY_SIMPLEX,

            # Font size
            0.6,

            # Text color (Green)
            (0, 255, 0),

            # Text thickness
            2
        )

    # --------------------------------------------------
    # FPS Calculation
    # --------------------------------------------------

    # Get the current time
    current_time = time.time()

    # Calculate Frames Per Second (FPS)
    fps = 1 / (current_time - prev_time)

    # Update previous time for the next frame
    prev_time = current_time

    # --------------------------------------------------
    # Display FPS
    # --------------------------------------------------

    # Display FPS on the top-left corner of the frame
    cv2.putText(

        frame,

        # Display FPS with one decimal place
        f"FPS : {fps:.1f}",

        # Position of the text
        (20, 35),

        # Font style
        cv2.FONT_HERSHEY_SIMPLEX,

        # Font size
        0.8,

        # Text color (Blue)
        (255, 0, 0),

        # Text thickness
        2
    )

    # --------------------------------------------------
    # Display Face Count
    # --------------------------------------------------

    # Display the total number of detected faces
    cv2.putText(

        frame,

        # Number of detected faces
        f"Faces : {len(faces)}",

        # Position of the text
        (20, 70),

        # Font style
        cv2.FONT_HERSHEY_SIMPLEX,

        # Font size
        0.8,

        # Text color (Red)
        (0, 0, 255),

        # Text thickness
        2
    )

    # Display the final output window
    cv2.imshow("Haar Cascade Face Detection", frame)

    # Wait for 1 millisecond
    # Exit the program when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------------------------------------
# Cleanup
# --------------------------------------------------

# Release the webcam resource
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()