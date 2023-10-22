import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = video.read()

    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imshow("video frame", frame)

      # setting values for base colors 
    b = frame[:, :, :1] 
    g = frame[:, :, 1:2] 
    r = frame[:, :, 2:] 
  
    # computing the mean 
    b_mean = np.mean(b) 
    g_mean = np.mean(g) 
    r_mean = np.mean(r) 
  
    # displaying the most prominent color 
    if (b_mean > g_mean and b_mean > r_mean): 
        print("Blue") 
    if (g_mean > r_mean and g_mean > b_mean): 
        print("Green") 
    else: 
        print("Red")

     # Break the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
    