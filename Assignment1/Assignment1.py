import cv2
import time
import numpy as np

cap = cv2.VideoCapture(1)
fps = 0 
while(True):
    tic = time.time()
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find the brightest spot using minMaxLoc
    _, _, _, brightest_loc = cv2.minMaxLoc(gray)
    
    # Mark the brightest spot on the original frame

    cv2.drawMarker(frame, brightest_loc, (0, 0, 255), markerType=cv2.MARKER_DIAMOND, thickness=1)
    #Brightest spot text
    cv2.putText(frame, f"Brightest", (brightest_loc[0]-40,brightest_loc[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Intensity values for all colors
    red_channel = frame[:, :, 2]
    green_channel = frame[:, :, 1]
    blue_channel = frame[:, :, 0]

    # Find reddest spot by finding max (red/blue+green) plus small number in denominator to avoid division with 0 
    color_ratio = red_channel/(green_channel + blue_channel+1e-5)
    color_ratio = color_ratio.astype(np.uint8)

    _, _, _, reddest_loc = cv2.minMaxLoc(color_ratio)

    cv2.drawMarker(frame, reddest_loc, (0, 0, 255), markerType=cv2.MARKER_TRIANGLE_UP, thickness=2)
    cv2.putText(frame, "Reddest", (reddest_loc[0] - 40, reddest_loc[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)



    #FPS text
    cv2.putText(frame, f"FPS = {fps:.0f}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('frame',frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    toc = time.time()
    fps = 1/(toc-tic)



cap.release()
cv2.destroyAllWindows()