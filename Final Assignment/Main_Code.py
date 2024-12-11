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
ball_diameter = 42.67e-3 #m 

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
    """
    Calculate the angle between two vectors (v1, v2).
    """
    # Compute dot product and magnitudes
    dot_product = sum(v1[i] * v2[i] for i in range(len(v1)))
    magnitude_v1 = math.sqrt(sum(v1[i]**2 for i in range(len(v1))))
    magnitude_v2 = math.sqrt(sum(v2[i]**2 for i in range(len(v1))))

    # Avoid division by zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0

    # Compute cosine of angle and clip it to the valid range [-1, 1]
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    cos_angle = max(min(cos_angle, 1.0), -1.0)

    # Calculate angle in radians and convert to degrees
    angle = math.acos(cos_angle)
    return math.degrees(angle)

def calculate_club_speed(ball_box_width, pixel_speed, standard_ball_diameter=42.67e-3, fps=240):
    """
    Convert pixel speed to real-world speed in m/s
    
    Parameters:
    - ball_box_width: Width of the ball's bounding box in pixels
    - pixel_speed: Speed calculated in pixels per second
    - standard_ball_diameter: Diameter of a standard golf ball in meters
    - fps: Frames per second of the video
    
    Returns:
    Real-world speed in meters per second
    """
    # Scale factor to convert pixels to meters
    # Use the standard ball diameter to calculate the real-world to pixel ratio
    pixels_per_meter = ball_box_width / standard_ball_diameter
    
    # Convert pixel speed to meters per second
    real_world_speed = pixel_speed / pixels_per_meter
    
    return real_world_speed


def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), BLACK, cv2.FILLED)
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def pre_process(input_image):
    """Preprocess the image for the model."""
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
    return blob

def post_process(input_image, outputs):
    """Post-process the model output."""
    global club_points
    class_ids = []
    confidences = []
    boxes = []
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        box = boxes[i]
        left, top, width, height = box
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3 * THICKNESS)
        label = f"{classes[class_ids[i]]}:{confidences[i]:.2f}"
        draw_label(input_image, label, left, top)

        # Track only the "Club"
        if classes[class_ids[i]] == "Club":
            # Calculate the center of the bounding box
            center_x = left + width // 2
            center_y = top + height // 2
            club_points.append((center_x, center_y))  # Add center point to the list

            # Limit the number of points stored
            if len(club_points) > 1400:  # Keep only the last 400 points
                club_points.pop(0)
    
    # Draw a continuous line connecting the club points
    if len(club_points) > 1:
        cv2.polylines(input_image, [np.array(club_points, np.int32)], isClosed=False, color=YELLOW, thickness=2)

    # COMPUTE 5 frame average club speed
    if len(club_points) > 5 and ball_box_width is not None: 
        speeds = []
        for i in range(1,5):
            last_point = club_points[-i]
            second_last_point = club_points[-i-1]
            
            # Calculate pixel distance
            pixel_distance = math.sqrt(
                (last_point[0] - second_last_point[0])**2 +
                (last_point[1] - second_last_point[1])**2
            )
            
            # Assuming 240 FPS, calculate speed in pixels per second
            pixel_speed = pixel_distance * 240
            real_world_speed = calculate_club_speed(ball_box_width,pixel_speed)
            speeds.append(real_world_speed)
        # Calculate average speed
        avg_club_speed = sum(speeds) / len(speeds)
        
        # Display average club speed on the image
        cv2.putText(input_image, f"Avg Club Speed: {avg_club_speed:.2f} m/s", 
                    (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)

    return input_image

def draw_pose_landmarks(frame, result):
    """Draw body pose landmarks and connections, excluding the face and replacing the side lines with a single central line."""
    if result.pose_landmarks:
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

        # Get coordinates for shoulders, hips, knees, elbows, and wrists
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
        left_elbow = [int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * width),
                      int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * height)]
        left_wrist = [int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * width),
                      int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * height)]
        right_foot = [int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * width),
                      int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * height)]
        left_foot = [int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * width),
                      int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * height)]
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

        # Calculate vectors for the thighs
        right_thigh_vector = [right_knee[0] - right_hip[0], right_knee[1] - right_hip[1]]
        left_thigh_vector = [left_knee[0] - left_hip[0], left_knee[1] - left_hip[1]]



        
        # Calculate vectors for the left arm
        left_arm_vector = [left_elbow[0] - left_shoulder[0], left_elbow[1] - left_shoulder[1]]

        # Calculate angles and determine colors
        thigh_angle = calculate_angle_between_vectors(right_thigh_vector, left_thigh_vector)
        thigh_color = GREEN if 25 <= thigh_angle <= 40 else RED
        foot_color = GREEN if 25 <= thigh_angle <= 40 else RED



        body_to_right_thigh_angle = calculate_angle_between_vectors(body_vector, right_thigh_vector)
        # body_to_right_thigh_color = GREEN if 30 <= body_to_right_thigh_angle <= 50 else RED

        body_to_left_arm_angle = calculate_angle_between_vectors(body_vector, left_arm_vector)
        left_arm_color = GREEN if 45 <= body_to_left_arm_angle <= 75 else RED

        # Display angles on the frame
        cv2.putText(frame, f"Leg angle: {int(thigh_angle)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, thigh_color, 2)
        # cv2.putText(frame, f"Body-R. Thigh: {int(body_to_right_thigh_angle)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, body_to_right_thigh_color, 2)
        # cv2.putText(frame, f"Body-L. Arm: {int(body_to_left_arm_angle)}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, left_arm_color, 2)
        # cv2.putText(frame, f"L. Thigh-Leg: {int(left_thigh_to_leg_angle)}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, left_leg_color, 2)

        # Draw lines with dynamic colors
        cv2.line(frame, tuple(right_hip), tuple(right_knee), thigh_color, 2)
        cv2.line(frame, tuple(left_hip), tuple(left_knee), thigh_color, 2)
        cv2.line(frame, tuple(right_knee), tuple(right_foot), thigh_color, 2)
        cv2.line(frame, tuple(left_knee), tuple(left_foot), thigh_color, 2)
        cv2.line(frame, tuple(mid_shoulder), tuple(mid_hip), GREEN, 2)  # Body line stays green
        cv2.line(frame, tuple(left_shoulder), tuple(left_elbow), left_arm_color, 2)
        # cv2.line(frame, tuple(left_hip), tuple(left_knee), left_leg_color, 2)


# Paths
model_path = "C:/Users/arnar/YOLOv5/models/ClubAndBall.onnx"

# Load class names
classes = ['Ball', 'Shaft', 'Club']


# Initialize ONNX Runtime session
ort_session = ort.InferenceSession(model_path)
input_name = ort_session.get_inputs()[0].name

# Open video capture
cap = cv2.VideoCapture("SlowSwing.MOV")
#cap = cv2.VideoCapture(1)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # Fixed at 30 FPS
output_path = "output_video.mp4"

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (frame_width//2, frame_height//2))

    # Pre-process the image
    blob = pre_process(frame)

    # Run inference
    outputs = ort_session.run(None, {input_name: blob})

    # Calculate and display FPS
    count += 1
    current_time = time.time()
    if current_time - start_time >= 1:  # Update FPS every second
        FPS_count = count / (current_time - start_time)
        count = 0
        start_time = current_time

    cv2.putText(frame, f"FPS: {FPS_count:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Post-process the outputs
    img = post_process(frame.copy(), outputs)

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    pose_result = pose.process(rgb_frame)

    # Draw pose landmarks and angle tracking
    draw_pose_landmarks(img, pose_result)

    # Write the processed frame to the output video
    out.write(img)

    # Display the frame
    cv2.imshow('Output', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()