# from inference_sdk import InferenceHTTPClient
import cv2
from inference import get_model

# Initialize the inference client
# 

model = get_model(model_id="golf_head_club_detect_v2/1")

# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="pagYRXfy4aKNH5AatqbQ"
# )

# Perform inference
mynd = "tiger.png"
# result = CLIENT.infer(mynd, model_id="golf_head_club_detect_v2/1")

# Read the image
image = cv2.imread(mynd)
result = model.infer(mynd)

# Class labels
class_labels = {
    0: "Shaft",
    1: "Head",
    2: "Hands"
}

# Draw bounding boxes on the image
for prediction in result["predictions"]:
    # Get bounding box details
    x = prediction["x"]
    y = prediction["y"]
    width = prediction["width"]
    height = prediction["height"]
    confidence = prediction["confidence"]
    class_id = prediction["class_id"]

    # Calculate box coordinates
    x1 = int(x - width / 2)
    y1 = int(y - height / 2)
    x2 = int(x + width / 2)
    y2 = int(y + height / 2)

    # Define color for bounding box (different colors for different classes)
    color = (0, 255, 0) if class_id == 0 else (0, 0, 255) if class_id == 1 else (255, 0, 0)

    # Draw the rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Add label and confidence
    label = f"{class_labels.get(class_id, 'Unknown')}: {confidence:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the image with bounding boxes
cv2.imshow("Golf Club Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
