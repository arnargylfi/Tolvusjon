import cv2
import numpy as np
import time

cap = cv2.VideoCapture(1)

def calculate_fps(prev_time):
    """
    Calculate frames per second
    """
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    return current_time, int(fps)

def are_lines_similar(line1, line2, angle_threshold=5, distance_threshold=40):
    """
    More strict line similarity checking.
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    # Compare angles more strictly
    angle_diff = abs(theta1 - theta2)
    angle_diff = min(angle_diff, np.pi - angle_diff)
    
    # Compare distances from origin more strictly
    distance_diff = abs(rho1 - rho2)
    
    return (angle_diff < angle_threshold) and (distance_diff < distance_threshold)

def find_intersection(line1, line2):
    """
    Finds the intersection of two lines in polar coordinates (rho, theta).
    Returns None if the lines are parallel or too similar.
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    # Convert polar to Cartesian coefficients: ax + by = c
    a1, b1, c1 = np.cos(theta1), np.sin(theta1), rho1
    a2, b2, c2 = np.cos(theta2), np.sin(theta2), rho2
    
    # Solve the linear equations
    determinant = a1 * b2 - a2 * b1
    
    # Check for near-parallel lines or small determinant
    if abs(determinant) < 1e-3:
        return None
    
    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant
    return int(x), int(y)

def order_points(pts):
    """
    Order points in top-left, top-right, bottom-right, bottom-left order
    """
    # Sort points based on x-coordinate
    xSorted = pts[np.argsort(pts[:, 0]), :]
    
    # Get left-most and right-most points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    
    # Sort left-most points vertically
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    
    # Sort right-most points vertically
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

# Destination image dimensions
height, width = 400, 300
pts_dst = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype=np.float32)

# Initialize FPS variables
prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    prev_time, fps = calculate_fps(prev_time)

    # Display FPS in the top-left corner
    fps_text = f"FPS: {fps}"
    cv2.putText(frame, fps_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 0), 2)

    # Convert frame to grayscale and detect edges with more conservative parameters
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Adjusted Canny parameters for less noise
    edges = cv2.Canny(gray, 100, 200)

    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)  # Reduced threshold
    intersections = []

    if lines is not None:
        # Filter out similar lines
        unique_lines = []
        for line in lines:
            # Stop if we've already found 4 unique lines
            if len(unique_lines) == 4:
                break
            
            is_unique = True
            for unique_line in unique_lines:
                if are_lines_similar(line[0], unique_line[0]):
                    is_unique = False
                    break
        
            if is_unique:
                unique_lines.append(line)
        # Draw lines and find intersections
        for i in range(len(unique_lines)):
            rho, theta = unique_lines[i][0]
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
            for j in range(i + 1, len(unique_lines)):
                intersection = find_intersection(unique_lines[i][0], unique_lines[j][0])
                if intersection:
                    intersections.append(intersection)
        
        # Draw intersection points
        for point in intersections:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Green dot
    
    # Check if we have at least 4 intersection points
    if len(intersections) > 4:
        # Use the first 4 intersection points as source points
        pts_src = np.array(intersections[:4], dtype=np.float32)
        
        try:
            # Order points correctly
            ordered_pts_src = order_points(pts_src)

            # Compute homography and warp the perspective
            tform, _ = cv2.findHomography(ordered_pts_src, pts_dst)
            warped = cv2.warpPerspective(frame, tform, (width, height))
            
            # Display the warped image in a separate window to prevent overlap
            cv2.imshow("Warped Image", warped)
        except Exception as e:
            print("Error in perspective transform:", e)
    
    # Display the original frame with lines and intersections
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()