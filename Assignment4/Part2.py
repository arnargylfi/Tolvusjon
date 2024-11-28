import cv2
import numpy as np

def find_intersection(line1, line2):
    """
    Finds the intersection of two lines in polar coordinates (rho, theta).
    Returns None if the lines are parallel.
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    # Convert polar to Cartesian coefficients: ax + by = c
    a1, b1, c1 = np.cos(theta1), np.sin(theta1), rho1
    a2, b2, c2 = np.cos(theta2), np.sin(theta2), rho2
    
    # Solve the linear equations
    determinant = a1 * b2 - a2 * b1
    if determinant == 0:
        return None  # Lines are parallel
    
    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant
    return int(x), int(y)

cap = cv2.VideoCapture(0)

# Destination image dimensions
height, width = 400, 300
pts_dst = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    intersections = []

    if lines is not None:
        # Draw lines and find intersections
        for i in range(len(lines)):
            rho, theta = lines[i][0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Draw lines on the frame
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Find intersections with other lines
            for j in range(i + 1, len(lines)):
                intersection = find_intersection(lines[i][0], lines[j][0])
                if intersection:
                    intersections.append(intersection)
        
        # Draw intersection points
        for point in intersections:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Green dot
    
    # Check if we have at least 4 intersection points
    if len(intersections) >= 4:
        # Use the first 4 intersection points as source points
        pts_src = np.array(intersections[:4], dtype=np.float32)

        # Compute homography and warp the perspective
        tform, _ = cv2.findHomography(pts_src, pts_dst)
        warped = cv2.warpPerspective(frame, tform, (width, height))
        
        # Display the warped image
        cv2.imshow("Warped Image", warped)
    
    # Display the original frame with lines and intersections
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
