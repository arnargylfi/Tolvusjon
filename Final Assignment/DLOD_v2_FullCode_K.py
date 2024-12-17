
import cv2

import numpy as np

import onnxruntime as ort

import mediapipe as mp

import time

import math

 

# Timing variables

start_time = time.time()

count = 0

FPS_count = 0

 

# Constants

INPUT_WIDTH = 640

INPUT_HEIGHT = 640

SCORE_THRESHOLD = 0.5

NMS_THRESHOLD = 0.45

CONFIDENCE_THRESHOLD = 0.45

 

# Text parameters

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX

FONT_SCALE = 0.7

THICKNESS = 1

 

# Colors

BLACK = (0, 0, 0)

BLUE = (255, 178, 50)

YELLOW = (0, 255, 255)

GREEN = (0, 255, 0)

RED = (0, 0, 255)

 

club_points = []

skip_points_clear = 0

ball_diameter = 42.67e-3

persistent_circle = None

save_image_time = None

current_window = None

last_time = 0     # Added for

last_last_x = 0

last_last_y = 0

last_x = 0

last_y = 0

last_radius_s = 0

 

# Initialize MediaPipe Pose

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5,

    static_image_mode=False,

    model_complexity=1,

    smooth_landmarks=True

)

 

def calculate_angle_between_vectors(v1, v2):

    dot_product = sum(v1[i] * v2[i] for i in range(len(v1)))

    magnitude_v1 = math.sqrt(sum(v1[i]**2 for i in range(len(v1))))

    magnitude_v2 = math.sqrt(sum(v2[i]**2 for i in range(len(v1))))

 

    if magnitude_v1 == 0 or magnitude_v2 == 0:

        return 0

 

    angle = math.acos(dot_product / (magnitude_v1 * magnitude_v2))

    return math.degrees(angle)

 

def smooth_trajectory(points, alpha=0.2):

    smoothed_points = []

    for i, (x, y) in enumerate(points):

        if i == 0:

            smoothed_points.append((x, y))  # First point remains unchanged

        else:

            prev_x, prev_y = smoothed_points[-1]

            new_x = alpha * x + (1 - alpha) * prev_x

            new_y = alpha * y + (1 - alpha) * prev_y

            smoothed_points.append((new_x, new_y))

    return smoothed_points

 

# Method to calculate circle intersection to obtain prediction for the club detection (location)

def circle_inters(h_c, k_c, d, h_s, k_s, r_s):

    # Step 1: Compute the distance between the centers

    D = math.sqrt((h_s - h_c)**2 + (k_s - k_c)**2)

 

    # Ensure the circles intersect

    if D > r_s + d or D < abs(r_s - d):

        raise ValueError("The circles do not intersect.")

 

    # Step 2: Distance along the line connecting the centers

    a = (d**2 - r_s**2 + D**2) / (2 * D)

 

    # Step 3: Midpoint of the chord

    u_x = (h_s - h_c) / D

    u_y = (k_s - k_c) / D

    m_x = h_c + a * u_x

    m_y = k_c + a * u_y

 

    # Step 4: Half-length of the chord

    h = math.sqrt(d**2 - a**2)

 

    # Step 5: Intersection points

    v_x = -u_y  # Perpendicular direction to the line connecting the centers

    v_y = u_x

    p1 = (m_x + h * v_x, m_y + h * v_y)

    p2 = (m_x - h * v_x, m_y - h * v_y)

 

    return [p1, p2]

 

def calculate_club_speed(ball_box_width, pixel_speed, standard_ball_diameter=42.67e-3, fps=240):

    # Scale factor to convert pixels to meters

    # Use the standard ball diameter to calculate the real-world to pixel ratio

    pixels_per_meter = ball_box_width / standard_ball_diameter

   

    # Convert pixel speed to meters per second

    real_world_speed = pixel_speed / pixels_per_meter

   

    return real_world_speed

 

def draw_label(im, label, x, y):

    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)

    dim, baseline = text_size[0], text_size[1]

    cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), BLACK, cv2.FILLED)

    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

 

def pre_process(input_image):

    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)

    return blob

 

Golf_ball_onoff = 1

 

def post_process(input_image, outputs, pose_result):

    global skip_points_clear, persistent_circle, save_image_time, club_points, current_window

    global last_last_x, last_last_y, last_x, last_y, last_time, last_radius_s

 

    # Initialize variables to avoid UnboundLocalError

    center_x = None

    center_y = None

    club_detected = False  # New flag to track if Club is detected

    right_shoulder = None  # Initialize right_shoulder

 

    live_video_frame = input_image.copy()

 

    class_ids = []

    confidences = []

    boxes = []

    rows = outputs[0].shape[1]

    image_height, image_width = input_image.shape[:2]

    x_factor = image_width / INPUT_WIDTH

    y_factor = image_height / INPUT_HEIGHT

 

    ball_box_width = None

 

    # Extract pose landmarks

    if pose_result.pose_landmarks:

        landmarks = pose_result.pose_landmarks.landmark

        right_ankle = [int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x*image_width),

                        int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y*image_height)]

        left_ankle = [int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x*image_width),

                        int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y*image_height)]

        right_hip = [int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x*image_width),

                        int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y*image_width)]

        left_hip = [int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x*image_width),

                    int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y*image_width)]

        right_shoulder = [int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x*image_width),

                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*image_height)]

        left_shoulder = [int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x*image_width),

                          int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y*image_height)]

    else:

        print("Pose landmarks not detected.")

 

    for r in range(rows):

        row = outputs[0][0][r]

        confidence = row[4]

        if confidence >= CONFIDENCE_THRESHOLD:

            classes_scores = row[5:]

            class_id = np.argmax(classes_scores)

            if classes_scores[class_id] > SCORE_THRESHOLD:

                confidences.append(confidence)

                class_ids.append(class_id)

                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w / 2) * x_factor)

                top = int((cy - h / 2) * y_factor)

                width = int(w * x_factor)

                height = int(h * y_factor)

                box = np.array([left, top, width, height])

                boxes.append(box)

                if classes[class_ids[-1]] == "Ball":

                    ball_box_width = width

                    print(f"ball box width detected:({ball_box_width})")

 

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for i in indices:

        box = boxes[i]

        left, top, width, height = box

        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3 * THICKNESS)

        label = f"{classes[class_ids[i]]}:{confidences[i]:.2f}"

        draw_label(input_image, label, left, top)

 

        # Check if the detected class is Club

        if classes[class_ids[i]] == "Club":

            club_detected = True  # Set flag if Club is detected

            center_x = left + width // 2

            center_y = top + height // 2

            club_points.append((center_x, center_y))

 

            last_last_x = last_x

            last_last_y = last_y

            last_x = center_x

            last_y = center_y

 

            if right_shoulder:  # Ensure right_shoulder is defined

                radius_s = math.sqrt((center_x - right_shoulder[0]) ** 2 + (center_y - right_shoulder[1]) ** 2)

                last_radius_s = radius_s

                last_time = time.time()

            print(f"Club detected at: ({center_x}, {center_y})")

 

    # Perform estimation only if Club is not detected

    if not club_detected:

        try:

            if right_shoulder and pose_result.pose_landmarks and len(club_points) > 2:

                # Use the last estimated point as the center of the smaller circle

                center_x = last_x

                center_y = last_y

 

                # Calculate the distance for the smaller circle (radius_c)

                dist = math.sqrt((last_last_x - last_x) ** 2 + (last_last_y - last_y) ** 2)

                radius_c = max(dist, 5)  # Ensure a minimum radius for estimation

 

                # Update the last radius of the shoulder if available

                if last_radius_s == 0:

                    last_radius_s = math.sqrt(

                        (center_x - right_shoulder[0]) ** 2 + (center_y - right_shoulder[1]) ** 2

                    )

 

                # Calculate intersection points

                print(f"Circle 1 Center: ({center_x}, {center_y}), Radius: {radius_c}")

                print(f"Circle 2 Center: ({right_shoulder[0]}, {right_shoulder[1]}), Radius: {last_radius_s}")

 

                intersect = circle_inters(center_x, center_y, radius_c, right_shoulder[0], right_shoulder[1], last_radius_s)

 

                # Choose the intersection point further from the last-last point

                distp1 = math.sqrt((intersect[0][0] - last_last_x) ** 2 + (intersect[0][1] - last_last_y) ** 2)

                distp2 = math.sqrt((intersect[1][0] - last_last_x) ** 2 + (intersect[1][1] - last_last_y) ** 2)

 

                if distp1 > distp2:

                    center_x, center_y = intersect[0]

                else:

                    center_x, center_y = intersect[1]

 

                # Update points

                club_points.append((center_x, center_y))

                last_last_x = last_x

                last_last_y = last_y

                last_x, last_y = center_x, center_y

 

                print(f"Estimated trajectory point: ({center_x}, {center_y})")

 

        except ValueError as e:

            print(f"Skipping frame due to error in circle_inters: {e}")

        except Exception as e:

            print(f"Unexpected error occurred: {e}")

 

    if len(club_points) > 800:

        club_points.pop(0)

 

    if len(club_points) > 1:

        smoothed_points = smooth_trajectory(club_points)

        cv2.polylines(live_video_frame, [np.array(smoothed_points, np.int32)], isClosed=False, color=YELLOW, thickness=2)

 

    if pose_result.pose_landmarks:

 

        reset_coordinate_x_left = right_ankle[0] + (left_ankle[0] - right_ankle[0]) / 3

        reset_coordinate_x_right = left_ankle[0] - (left_ankle[0] - right_ankle[0]) / 3

       

        if center_x is not None and center_y is not None:

            if reset_coordinate_x_left < center_x < reset_coordinate_x_right and center_y > max(right_ankle[1], left_ankle[1]) and skip_points_clear == 0:

                club_points.clear()

 

            if center_x < right_ankle[0] and center_y < right_hip[1] and skip_points_clear == 0:

                skip_points_clear = 1

                persistent_circle = (center_x, center_y)

 

            if center_x > left_ankle[0] and center_y < left_hip[1] and skip_points_clear == 1:

                skip_points_clear = 0

                persistent_circle = None

                save_image_time = time.time()

 

    #return live_video_frame

 

    if save_image_time and time.time() - save_image_time >= 1:

        trajectory_image = input_image.copy()

        if len(club_points) > 1:

            cv2.polylines(trajectory_image, [np.array(club_points, np.int32)], isClosed=False, color=YELLOW, thickness=2)

        filename = f"trajectory_{int(time.time())}.jpg"

        cv2.imwrite(filename, trajectory_image)

        print(f"Saved trajectory image as {filename}")

 

        if current_window:

            cv2.destroyWindow(current_window)

        current_window = f"Trajectory {int(time.time())}"

        cv2.imshow(current_window, trajectory_image)

        cv2.moveWindow(current_window, 700, 30)

        cv2.waitKey(1)

 

        save_image_time = None

        club_points.clear()

 

    if persistent_circle is not None:

        cv2.circle(live_video_frame, persistent_circle, radius=10, color=(255, 0, 0), thickness=-1)

 

    if Golf_ball_onoff == 1 and ball_box_width is not None and len(club_points) > 5:

            speeds = []

            for i in range(1, 5):

                last_point = club_points[-i]

                second_last_point = club_points[-i - 1]

 

                pixel_distance = math.sqrt(

                    (last_point[0] - second_last_point[0])**2 +

                    (last_point[1] - second_last_point[1])**2

                )

 

                pixel_speed = pixel_distance * 240   # Ath breyta hér fyrir FPS

                real_world_speed = calculate_club_speed(ball_box_width, pixel_speed)

                speeds.append(real_world_speed)

 

            avg_club_speed = sum(speeds) / len(speeds)

 

            cv2.putText(live_video_frame, f"Avg Club Speed: {avg_club_speed:.2f} m/s",

                        (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)

 

    return live_video_frame

 

def draw_pose_landmarks(frame, result):

        landmarks = result.pose_landmarks.landmark

        height, width, _ = frame.shape

 

        # Draw all body connections excluding face and side lines between shoulders and hips

        for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:

            if start_idx >= 11 and end_idx >= 11:  # Skip face landmarks (indices < 11)

                if not (

                    (start_idx in [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_SHOULDER]

                     and end_idx in [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_HIP])

                    or

                    (start_idx in [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_HIP]

                     and end_idx in [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_SHOULDER])

                ):

                    start = landmarks[start_idx]

                    end = landmarks[end_idx]

 

                    # Convert normalized coordinates to pixel coordinates

                    start_point = (int(start.x * width), int(start.y * height))

                    end_point = (int(end.x * width), int(end.y * height))

 

                    # Draw the connection

                    cv2.line(frame, start_point, end_point, GREEN, 2)

 

        # Highlight joints (landmarks) except for face

        for i, landmark in enumerate(landmarks):

            if i >= 11:  # Skip face landmarks (indices < 11)

                x, y = int(landmark.x * width), int(landmark.y * height)

                cv2.circle(frame, (x, y), 5, RED, -1)

 

        # Get coordinates for key joints

        right_shoulder = [int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width),

                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)]

        left_shoulder = [int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width),

                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)]

        right_hip = [int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * width),

                     int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * height)]

        left_hip = [int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * width),

                    int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * height)]

        right_knee = [int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * width),

                      int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * height)]

        left_knee = [int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * width),

                     int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * height)]

        right_ankle = [int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * width),

                       int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * height)]

        left_ankle = [int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * width),

                      int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * height)]

        left_elbow = [int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * width),

                      int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * height)]

        left_wrist = [int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * width),

                      int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * height)]

 

        # Calculate midpoints for shoulders and hips

        mid_shoulder = [(right_shoulder[0] + left_shoulder[0]) // 2,

                        (right_shoulder[1] + left_shoulder[1]) // 2]

        mid_hip = [(right_hip[0] + left_hip[0]) // 2,

                   (right_hip[1] + left_hip[1]) // 2]

 

        # Draw a single line connecting midpoints of shoulders and hips

        cv2.line(frame, tuple(mid_shoulder), tuple(mid_hip), GREEN, 2)

 

        # Highlight joints (landmarks) used for the line

        cv2.circle(frame, tuple(mid_shoulder), 5, RED, -1)

        cv2.circle(frame, tuple(mid_hip), 5, RED, -1)

 

        # Calculate vector for the body line (shoulder to hip)

        body_vector = [mid_hip[0] - mid_shoulder[0], mid_hip[1] - mid_shoulder[1]]

 

        # Calculate vectors for the thighs and legs

        right_thigh_vector = [right_knee[0] - right_hip[0], right_knee[1] - right_hip[1]]

        left_thigh_vector = [left_knee[0] - left_hip[0], left_knee[1] - left_hip[1]]

 

        right_leg_vector = [right_ankle[0] - right_knee[0], right_ankle[1] - right_knee[1]]

        left_leg_vector = [left_ankle[0] - left_knee[0], left_ankle[1] - left_knee[1]]

 

        left_arm_vector = [left_elbow[0] - left_shoulder[0], left_elbow[1] - left_shoulder[1]]

 

        # Calculate angles and determine colors

        thigh_angle = calculate_angle_between_vectors(right_thigh_vector, left_thigh_vector)

        thigh_color = GREEN if 25 <= thigh_angle <= 40 else RED

 

        body_to_right_thigh_angle = calculate_angle_between_vectors(body_vector, right_thigh_vector)

        body_to_right_thigh_color = GREEN if 30 <= body_to_right_thigh_angle <= 50 else RED

 

        body_to_left_arm_angle = calculate_angle_between_vectors(body_vector, left_arm_vector)

        left_arm_color = GREEN if 45 <= body_to_left_arm_angle <= 75 else RED

 

        left_thigh_to_leg_angle = calculate_angle_between_vectors(left_thigh_vector, left_leg_vector)

        left_leg_color = GREEN if 10 <= left_thigh_to_leg_angle <= 30 else RED

 

        # Display angles on the frame

        cv2.putText(frame, f"Thigh Angle: {int(thigh_angle)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, thigh_color, 2)

        cv2.putText(frame, f"Body-R. Thigh: {int(body_to_right_thigh_angle)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, body_to_right_thigh_color, 2)

        cv2.putText(frame, f"Body-L. Arm: {int(body_to_left_arm_angle)}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, left_arm_color, 2)

        cv2.putText(frame, f"L. Thigh-Leg: {int(left_thigh_to_leg_angle)}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, left_leg_color, 2)

 

        # Draw lines with dynamic colors

        cv2.line(frame, tuple(right_hip), tuple(right_knee), thigh_color, 2)

        cv2.line(frame, tuple(left_hip), tuple(left_knee), thigh_color, 2)

        cv2.line(frame, tuple(mid_shoulder), tuple(mid_hip), body_to_right_thigh_color, 2)  # Body line stays green

        cv2.line(frame, tuple(left_shoulder), tuple(left_elbow), left_arm_color, 2)

        cv2.line(frame, tuple(left_knee), tuple(left_ankle), left_leg_color, 2)

 

# Paths

model_path = "C:/Users/kristinnh/Documents/Assignment 1/YOLOv5/ClubAndBall.onnx"

classes = ['Ball', 'Shaft', 'Club']

 

ort_session = ort.InferenceSession(model_path)

input_name = ort_session.get_inputs()[0].name

 

#cap = cv2.VideoCapture(1)

cap = cv2.VideoCapture("C:/Users/kristinnh/Desktop/Myndbönd-Computervision/IMG_0285.MOV")

 

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.7)

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.7)

#frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

#frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = 30

output_path = "C:/Users/kristinnh/Desktop/start_stop2.mp4"

 

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

 

while True:

    ret, frame = cap.read()

    if not ret:

        break

 

    frame = cv2.resize(frame, (frame_width, frame_height))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_result = pose.process(rgb_frame)

 

    blob = pre_process(frame)

    outputs = ort_session.run(None, {input_name: blob})

 

    img = post_process(frame.copy(), outputs, pose_result)

 

    count += 1

    current_time = time.time()

    if current_time - start_time >= 1:

        FPS_count = count / (current_time - start_time)

        count = 0

        start_time = current_time

 

    cv2.putText(img, f"FPS: {FPS_count:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    draw_pose_landmarks(img, pose_result)

    out.write(img)

    cv2.imshow('Output', img)

 

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

 

cap.release()

out.release()

cv2.destroyAllWindows()
