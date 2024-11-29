import cv2
import numpy as np
import itertools
import time

# Parameters
t_lower = 100  # Lower threshold for Canny
t_upper = 200  # Upper threshold for Canny
rho = 1  # Distance resolution in pixels of the Hough grid
theta = np.pi / 180  # Angular resolution in radians of the Hough grid
threshold = 50  # Minimum number of votes to consider a line
min_line_length = 100  # Minimum length of a line (in pixels)
max_line_gap = 5  # Maximum allowed gap between line segments to treat them as a single line

# Timing variables for FPS
start_time = time.time()
count = 0
FPS_count = 0

cap = cv2.VideoCapture(1)

# Resize scale factor
resize_factor = 1  # Resize as needed (1 equals 100%)

def is_crossed_quadrangle(points):
    #Check if a quadrangle is crossed by detecting self-intersecting edges
    def do_intersect(p1, q1, p2, q2):
        #Check if line segments p1q1 and p2q2 intersect
        def orientation(a, b, c):
            #Find orientation of ordered triplet (a, b, c)
            val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
            if val == 0:
                return 0  # Collinear
            return 1 if val > 0 else 2  # Clockwise or Counterclockwise

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special Cases
        return False

    # Check all pairs of edges
    for i in range(4):
        p1, q1 = points[i], points[(i + 1) % 4]
        for j in range(i + 2, 4):
            p2, q2 = points[j], points[(j + 1) % 4]
            if (i == 0 and j == 3):  # Avoid checking adjacent edges
                continue
            if do_intersect(p1, q1, p2, q2):
                return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

    # Preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, t_lower, t_upper)

    # Apply Hough Transform to find lines
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is not None:
        # Find intersections of the lines
        def line_intersection(line1, line2):
            """Calculate the intersection point of two lines."""
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2

            # Solve linear equations to find the intersection
            A1 = y2 - y1
            B1 = x1 - x2
            C1 = A1 * x1 + B1 * y1

            A2 = y4 - y3
            B2 = x3 - x4
            C2 = A2 * x3 + B2 * y3

            determinant = A1 * B2 - A2 * B1

            if determinant == 0:
                return None  # Lines are parallel
            else:
                x = (B2 * C1 - B1 * C2) / determinant
                y = (A1 * C2 - A2 * C1) / determinant
                return int(x), int(y)

        # Get all combinations of lines to find intersections
        intersections = []
        for line1, line2 in itertools.combinations(lines[:, 0], 2):
            pt = line_intersection(line1, line2)
            if pt and 0 <= pt[0] < frame.shape[1] and 0 <= pt[1] < frame.shape[0]:
                intersections.append(pt)

        if len(intersections) >= 4:
            intersections = np.array(intersections)

            # Find contours and approximate as polygons
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # Approximate contour with accuracy proportional to the contour perimeter
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:  # Four-sided shape detected
                    points = approx.reshape(4, 2)
                    if is_crossed_quadrangle(points):
                        continue  # Skip crossed quadrangles

                    # Draw the quadrangle
                    cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

                    # Perspective transformation
                    # src_pts = points.astype("float32")
                    # dst_pts = np.array([[0, 0], [400 - 1, 0], [400 - 1, 300 - 1], [0, 300 - 1]], dtype="float32")
                    # matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

                    # Warp the image
                    # warped = cv2.warpPerspective(frame, matrix, (400, 300))
                    # if warped is not None and warped.size > 0:
                    #     cv2.imshow("Rectified Shape", warped)
                    
                    # Perspective transformation
                    src_pts = points.astype("float32")
                    frame_height, frame_width = frame.shape[:2]  # Get original frame dimensions
                    dst_pts = np.array([[0, 0], [frame_width - 1, 0], [frame_width - 1, frame_height - 1], [0, frame_height - 1]], dtype="float32")
                    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

                    # Warp the image to the size of the original frame
                    warped = cv2.warpPerspective(frame, matrix, (frame_width, frame_height))

                    if warped is not None and warped.size > 0:
                        # Resize warped image to match the original frame's size
                        warped_resized = cv2.resize(warped, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                        cv2.imshow("Rectified Shape", warped_resized)




     # Calculate and display FPS
    count += 1
    current_time = time.time()
    if current_time - start_time >= 1:  # Update FPS every second
        FPS_count = count / (current_time - start_time)
        count = 0
        start_time = current_time

    cv2.putText(frame, f"FPS: {FPS_count:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) 
    
    # Display original frame with quadrangle overlay
    cv2.imshow("Detected Quadrangle", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()