import cv2
import numpy as np
import onnxruntime as ort
import os

# Constants
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Colors
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# Create output directory if it doesn't exist
# os.makedirs('detected_objects', exist_ok=True)
# os.makedirs('tracked_objects', exist_ok=True)

class ObjectTracker:
    def __init__(self, template, confidence, class_name):
        """
        Initialize object tracker with template image
        
        Parameters:
        - template: Detected object image
        - confidence: Detection confidence
        - class_name: Class of the detected object
        """
        # Ensure template is grayscale
        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        self.confidence = confidence
        self.class_name = class_name
        self.tracked_locations = []

    def update_template(self, frame, top_left, width, height):
        """
        Update the template using the most recent detection.
        
        Parameters:
        - frame: Current video frame
        - top_left: Top-left corner of the detected region
        - width: Width of the detected region
        - height: Height of the detected region
        """
        updated_template = frame[top_left[1]:top_left[1] + height, top_left[0]:top_left[0] + width]
        if len(updated_template.shape) == 3:
            updated_template = cv2.cvtColor(updated_template, cv2.COLOR_BGR2GRAY)
        self.template = updated_template

    def track(self, frame, method=cv2.TM_CCOEFF_NORMED):
        """
        Track object in the current frame using template matching
        
        Parameters:
        - frame: Current video frame (grayscale)
        - method: OpenCV template matching method
        
        Returns:
        - Best match location and confidence
        """
        # Ensure frame is grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Define the search area (ROI) around the last known location
        if self.tracked_locations:
            last_location = self.tracked_locations[-1]['location']
            search_margin = 75  # Pixels around the last location

            x1 = max(0, last_location[0] - search_margin)
            y1 = max(0, last_location[1] - search_margin)
            x2 = min(frame.shape[1], last_location[0] + self.template.shape[1] + search_margin)
            y2 = min(frame.shape[0], last_location[1] + self.template.shape[0] + search_margin)

            roi = frame[y1:y2, x1:x2]
        else:
            # If no previous location, use the entire frame
            roi = frame
            x1, y1 = 0, 0

        # Ensure template size is not larger than the ROI
        if self.template.shape[0] > roi.shape[0] or self.template.shape[1] > roi.shape[1]:
            scale = min(roi.shape[0] / self.template.shape[0], roi.shape[1] / self.template.shape[1])
            new_size = (int(self.template.shape[1] * scale), int(self.template.shape[0] * scale))
            self.template = cv2.resize(self.template, new_size)

        # Perform template matching within the ROI
        result = cv2.matchTemplate(roi, self.template, method)

        # Find the best match location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Adjust coordinates to the full frame
        top_left = (max_loc[0] + x1, max_loc[1] + y1)
        h, w = self.template.shape[:2]

        # Update the template with the new detection
        self.update_template(frame, top_left, w, h)

        # Store tracking information
        self.tracked_locations.append({
            'location': top_left,
            'confidence': max_val,
            'timestamp': len(self.tracked_locations)
        })

        return top_left, max_val, (w, h)

def detect_and_track_objects(input_image, ort_session, classes):
    """
    Detect objects in the first frame and set up trackers

    Parameters:
    - input_image: Input frame to process
    - ort_session: ONNX Runtime inference session
    - classes: List of class names

    Returns:
    - Processed image with bounding boxes
    - List of object trackers
    """
    # Pre-process the image
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)

    # Run inference
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: blob})

    # Image dimensions
    image_height, image_width = input_image.shape[:2]
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    # Storage for detected objects and trackers
    trackers = []
    class_ids = []
    confidences = []
    boxes = []

    # Process outputs
    rows = outputs[0].shape[1]
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]

        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)

            if classes_scores[class_id] > SCORE_THRESHOLD and classes[class_id] == "Club":
                confidences.append(confidence)
                class_ids.append(class_id)

                # Convert bounding box
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = [left, top, width, height]
                boxes.append(box)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Save and create trackers for detected objects
    for i in indices:
        box = boxes[i]
        left, top, width, height = box
        class_name = classes[class_ids[i]]
        confidence = confidences[i]

        # Extract the object
        object_img = input_image[top:top+height, left:left+width]

        # Save the object image
        # template_filename = f'detected_objects/{class_name}_{i}_conf_{confidence:.2f}.jpg'
        # cv2.imwrite(template_filename, object_img)

        # Create tracker for the object
        tracker = ObjectTracker(object_img, confidence, class_name)
        trackers.append((tracker, (left, top, width, height)))

        # Draw initial bounding box
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 2)

        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(input_image, label, (left, top - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2)

    return input_image, trackers
def main():
    # Paths
    model_path = "C:/Users/arnar/YOLOv5/models/ClubAndBall.onnx"
    video_path = "SlowSwing.MOV"
    
    # Load class names
    classes = ['Ball', 'Shaft', 'Club']
    
    # Initialize ONNX Runtime session
    ort_session = ort.InferenceSession(model_path)
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('tracked_objects_video.mp4', fourcc, fps, (frame_width, frame_height))
    
    # Process first frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
    
    # Resize the frame
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    
    # Detect and create trackers for objects
    processed_frame, trackers = detect_and_track_objects(frame, ort_session, classes)
    
    # Track objects in subsequent frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        
        # Track each object
        for tracker, (initial_left, initial_top, initial_width, initial_height) in trackers:
            # Track the object
            (top_left, confidence, (w, h)) = tracker.track(frame)
            
            # Draw tracking results
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame, top_left, bottom_right, GREEN, 2)
            
            # Add tracking confidence
            label = f"{tracker.class_name}: {confidence:.2f}"
            cv2.putText(frame, label, top_left, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
            
            # Save tracked object images periodically
            # if frame_count % 10 == 0:
            #     tracked_obj = frame[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
                # cv2.imwrite(f'tracked_objects/{tracker.class_name}_frame_{frame_count}.jpg', tracked_obj)
        
        # Write the frame to output video
        out.write(frame)
        
        # Display the frame
        cv2.imshow('Object Tracking', frame)
        
        # Increment frame counter
        frame_count += 1
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()