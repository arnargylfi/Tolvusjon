from inference_sdk import InferenceHTTPClient
import cv2

def main():
    # Initialize the inference client
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="pagYRXfy4aKNH5AatqbQ"
    )

    # Open webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Class labels
    class_labels = {
        0: "Shaft",
        1: "Head", 
        2: "Hands"
    }

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Save the captured frame temporarily

        # Perform inference on the captured frame
        result = CLIENT.infer(frame, model_id="golf_head_club_detect_v2/1")

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add label and confidence
            label = f"{class_labels.get(class_id, 'Unknown')}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the resulting frame
        cv2.imshow("Golf Club Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()