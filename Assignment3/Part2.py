import os
import glob
import matplotlib.pyplot as plt
import cv2
import requests
import zipfile

# Constants
TRAIN = True
EPOCHS = 25
base_dir = "C:/Users/arnar/YOLOv5"
data_yaml = f"{base_dir}/data.yaml"
dirs = ['train', 'valid', 'test']


# Set the result directory name
def set_res_dir():
    return f"experiment_{EPOCHS}_epochs"

RES_DIR = set_res_dir()

# Change to YOLOv5 directory
os.chdir(base_dir)

# Train the YOLOv5 model
train_command = f'python train.py --data "{data_yaml}" --weights yolov5s.pt --img 640 --epochs {EPOCHS} --batch-size 16 --name "{RES_DIR}"'
print(f"Running training command: {train_command}")
os.system(train_command)

# Function to show validation predictions saved during training.
def show_valid_results(RES_DIR):
    exp_path = f"runs/train/{RES_DIR}"
    validation_pred_images = glob.glob(f"{exp_path}/*_pred.jpg")
    if validation_pred_images:
        print(validation_pred_images)
        for pred_image in validation_pred_images:
            image = cv2.imread(pred_image)
            plt.figure(figsize=(19, 16))
            plt.imshow(image[:, :, ::-1])
            plt.axis('off')
            plt.show()
    else:
        print(f"No validation prediction images found in {exp_path}")

# Helper function for inference on images.
def inference(RES_DIR, data_path):
    # Directory to store inference results.
    infer_dir_count = len(glob.glob('runs/detect/*'))
    print(f"Current number of inference detection directories: {infer_dir_count}")
    INFER_DIR = f"inference_{infer_dir_count+1}"
    print(f"Inference directory: {INFER_DIR}")

    # Run inference with YOLOv5 model on the test images.
    inference_command = f'python detect.py --weights "runs/train/{RES_DIR}/weights/best.pt" --source "{data_path}" --name "{INFER_DIR}"'
    print(f"Running inference command: {inference_command}")
    os.system(inference_command)
    return INFER_DIR

# Visualize inference results.
def visualize(INFER_DIR):
    INFER_PATH = f"runs/detect/{INFER_DIR}"
    infer_images = glob.glob(f"{INFER_PATH}/*.jpg")
    if infer_images:
        print(infer_images)
        for pred_image in infer_images:
            image = cv2.imread(pred_image)
            plt.figure(figsize=(19, 16))
            plt.imshow(image[:, :, ::-1])
            plt.axis('off')
            plt.show()
    else:
        print(f"No inference images found in {INFER_PATH}")

# Run inference on images.
IMAGE_INFER_DIR = inference(RES_DIR, f"{base_dir}/inference_images")
visualize(IMAGE_INFER_DIR)