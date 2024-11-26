import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)

# LOOKING AT THE PROCESSING TIME FOR SIMPLY DISPLAYING AN IMAGE FOR 5 SECONDS
def f1(runningtime):
    loops = 0
    timenow = 0
    begintime = time.time()
    while(timenow-begintime < runningtime):
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('frame',frame)
        timenow = time.time()
        loops+=1
    print(f"Displayed {loops} video frames in {timenow-begintime:.2f} seconds or {(timenow-begintime)/loops:.3f} seconds processing time per image")

# PROCESSING TIME WITH BRIGHTEST SPOT
def f2(runningtime):
    loops = 0
    timenow = 0
    begintime = time.time()
    while(timenow-begintime < runningtime):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find the brightest spot using minMaxLoc
        _, _, _, brightest_loc = cv2.minMaxLoc(gray)
    
        # Mark the brightest spot on the original frame
        cv2.circle(frame, brightest_loc,10, (255, 0, 0), thickness=1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('Marking brightest spot with inbuilt function',frame)
        timenow = time.time()
        loops+=1
    print(f"Displayed {loops} video frames with marking for brightest spot in {timenow-begintime:.2f} seconds or {(timenow-begintime)/loops:.3f} seconds processing time per loop iteration")

#IS THE PROCESSING TIME IDENTICAL WHEN NOT DISPLAYING IMAGE?
def f3(runningtime):
    loops = 0
    timenow = 0
    begintime = time.time()
    while(timenow-begintime < runningtime):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find the brightest spot using minMaxLoc
        _, _, _, brightest_loc = cv2.minMaxLoc(gray)
    
        # Mark the brightest spot on the original frame
        cv2.circle(frame, brightest_loc,10, (255, 0, 0), thickness=1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        timenow = time.time()
        loops+=1
    print(f"Processed {loops} video frames with marking for brightest spot, without displaying it in {timenow-begintime:.2f} seconds or {(timenow-begintime)/loops:.3f} seconds processing time per loop iteration")


#USING FOR LOOP INSTEAD OF BUILT IN FUNCTION TO FIND BRIGHT SPOT
def f4(runningtime):
    loops = 0
    timenow = 0
    begintime = time.time()
    while(timenow-begintime < runningtime):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find the brightest spot by iterating through each pixel 
        brightest_loc = (0, 0)
        max_intensity = -1
        for y in range(gray.shape[0]):  # Loop over rows (height)
            for x in range(gray.shape[1]):  # Loop over columns (width)
                pixel_intensity = gray[y, x]
                if pixel_intensity > max_intensity:
                    max_intensity = pixel_intensity
                    brightest_loc = (x, y)
    
        # Mark the brightest spot on the original frame
        cv2.circle(frame, brightest_loc,10, (255, 0, 0), thickness=1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('Brightest spot with double for loop',frame)
        timenow = time.time()
        loops+=1
    print(f"Displayed {loops} video frames with marking for brightest spot found with double for loop in {timenow-begintime:.2f} seconds or {(timenow-begintime)/loops:.3f} seconds processing time per loop iteration")


T = 10
f1(T)
f2(T)
f3(T)
f4(T)


cap.release()
cv2.destroyAllWindows()