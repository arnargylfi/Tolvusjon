import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection on grayscale image
    edges = cv2.Canny(gray, 50, 50)
    
    # Apply Hough transform to detect lines on the grayscale edges
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    # Draw the first 4 most prominent lines if any are found
    if lines is not None:
        for i in range(min(4, len(lines))):
            rho, theta = lines[i][0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            
            # Draw lines on the original color frame
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv2.imshow('frame', frame)
    cv2.imshow('edges', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()