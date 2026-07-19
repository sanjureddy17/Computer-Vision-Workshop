import cv2

# Open the default webcam (0 = primary webcam)
cap = cv2.VideoCapture(0)

# FOURCC = Four Character Code
# It specifies the video codec (compression format) used to save the video.

# Create a FOURCC codec for MP4 video format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 video encoder

# Create a VideoWriter object to save the recorded video
# Parameters:
# 'output.mp4' -> Output file name
# fourcc       -> Video codec
# 20.0         -> Frames Per Second (FPS)
# (640, 480)   -> Frame resolution (Width, Height)
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))



# Start an infinite loop to continuously capture webcam frames
while True:

    # Read a frame from the webcam
    # ret   -> True if frame is captured successfully
    # frame -> Captured image/frame
    ret, frame = cap.read()

    # If frame capture fails, exit the loop
    if not ret:
        break

    # Write the current frame to the output video file
    out.write(frame)

    # Display the live webcam feed in a window named "Recording"
    cv2.imshow('Recording', frame)

    # Wait for 1 millisecond for a key press
    # If the user presses 'q', exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam resource
cap.release()

# Release the video writer and save the video file
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()