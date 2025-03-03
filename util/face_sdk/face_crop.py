import glob
import logging.config
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple, List

import cv2
import ffmpeg
import yaml
from numpy import ndarray
from tqdm.auto import tqdm
import numpy as np

from marlin_pytorch.util import crop_with_padding
from util.face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from util.face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader

logging.config.fileConfig(os.path.join("util", "face_sdk", "config", "logging.conf"))
logger = logging.getLogger('api')

with open(os.path.join("util", "face_sdk", "config", "model_conf.yaml")) as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)

# common setting for all model, need not modify.
model_path = os.path.join("util", "face_sdk", 'models')

# model setting, modified along with model
scene = 'non-mask'
model_category = 'face_detection'
model_name = model_conf[scene][model_category]

logger.info('Start to load the face detection model...')
# load model
sys.path.append(os.path.join("util", "face_sdk"))
print("path of face_sdk is appended.")
faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
model, cfg = faceDetModelLoader.load_model()
faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)
print("Face detection model loaded")

def crop_face(frame, margin=1, x=0, y=0) -> Tuple[ndarray, int, int, int]:
    assert frame.ndim == 3 and frame.shape[2] == 3, "frame should be 3-dim"
    dets = faceDetModelHandler.inference_on_image(frame)
    if len(dets) > 0:
        x1, y1, x2, y2, confidence = dets[0]
        # center
        x, y = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        margin = int(max(abs(x2 - x1), abs(y2 - y1)) / 2)
    # crop face
    face = crop_with_padding(frame, x - margin, x + margin, y - margin, y + margin, 0)
    face = cv2.resize(face, (224, 224))
    return face, margin, x, y


def crop_face_video(video_path: str, save_path: str, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=30) -> None:
    cap = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter(save_path, fourcc=fourcc, fps=fps, frameSize=(224, 224))
    x, y = 0, 0
    margin = 1

    while True:
        ret, frame = cap.read()
        if ret:
            face, margin, x, y = crop_face(frame, margin, x, y)
            writer.write(face)
        else:
            break

    cap.release()
    writer.release()


def crop_face_img(img_path: str, save_path: str):
    frame = cv2.imread(img_path)
    if frame is None:
        print("something is wrong.")
    face = crop_face(frame)[0]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, face)




def process_videos(video_path, output_path, ext="mp4", max_workers=8):
    if ext.lower() == "mp4":  # Use lower() for case insensitive check
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif ext.lower() == "avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    elif ext.lower() == "wmv":
        fourcc = cv2.VideoWriter_fourcc(*"WMV1")  # You can also use "WMV2" depending on your needs
    else:
        raise ValueError("ext should be mp4, avi, or wmv")

    Path(output_path).mkdir(parents=True, exist_ok=True)

    files = os.listdir(video_path)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for f_name in tqdm(files):
            if f_name.endswith('.' + ext) or f_name.endswith('.' + ext.upper()):
                print(f"Processing {f_name} in face cropping")
                source_path = os.path.join(video_path, f_name)
                target_path = os.path.join(output_path, f_name)
                fps = eval(ffmpeg.probe(source_path)["streams"][0]["avg_frame_rate"])
                futures.append(executor.submit(crop_face_video, source_path, target_path, fourcc,
                    fps))
            else:
                print(f"the extention of the file {f_name} is not {ext}")
        for future in tqdm(futures):
            future.result()


def process_images(image_path: str, output_path: str, max_workers: int = 8):
    print("Processing images in face cropping")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    files = glob.glob(f"{image_path}/**/*.jpg", recursive=True)
    # files = glob.glob(f"{image_path}/*/*/*.jpg")
    print(f"there are {len(files)} images")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for file in tqdm(files, desc="Processing images:crop_face_img"):
            save_path = file.replace(image_path, output_path)
            Path("/".join(save_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
            futures.append(executor.submit(crop_face_img, file, save_path))

        for future in tqdm(futures):
            future.result()

            
def process_single_image_set(image_set: tuple) -> None:
    texture_path, depth_path, thermal_path, save_dir = image_set

    # Check if all modalities are present
    if not all([os.path.exists(texture_path), os.path.exists(depth_path), os.path.exists(thermal_path)]):
        return

    # Read the texture image
    texture_frame = cv2.imread(texture_path)
    if texture_frame is None:
        return
    
    # Get texture frame dimensions
    texture_h, texture_w = texture_frame.shape[:2]
    
    # Crop the texture image and get coordinates and margin
    face, margin, x, y = crop_face(texture_frame)

    # Get the directory of the texture image
    texture_dir = os.path.dirname(texture_path)

    # Calculate the relative path from the texture directory to the save directory
    relative_path = os.path.relpath(texture_dir, save_dir)

    # Construct save paths
    texture_save_path = os.path.join(save_dir, "Texture_crop_crop_images_DB", relative_path, os.path.basename(texture_path))
    depth_save_path = os.path.join(save_dir, "Depth_crop_crop_images_DB", relative_path, os.path.basename(depth_path))
    thermal_save_path = os.path.join(save_dir, "Thermal_crop_crop_images_DB", relative_path, os.path.basename(thermal_path))

    # Ensure the directories exist for saving cropped images
    os.makedirs(os.path.dirname(texture_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(depth_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(thermal_save_path), exist_ok=True)

    # Save cropped texture image
    cv2.imwrite(texture_save_path, face)

    # Crop depth image using scaled coordinates and margin
    depth_frame = cv2.imread(depth_path)
    if depth_frame is not None:
        depth_h, depth_w = depth_frame.shape[:2]
        
        # Scale coordinates and margin based on depth image dimensions
        depth_x = int(x * (depth_w / texture_w))
        depth_y = int(y * (depth_h / texture_h))
        depth_margin = int(margin * (depth_w / texture_w))  # Assuming aspect ratio is preserved
        
        # Ensure the coordinates are within the bounds of the depth image
        x_min = max(0, depth_x - depth_margin)
        x_max = min(depth_w, depth_x + depth_margin)
        y_min = max(0, depth_y - depth_margin)
        y_max = min(depth_h, depth_y + depth_margin)
        
        # Crop the depth image
        depth_face = depth_frame[y_min:y_max, x_min:x_max]
        
        # Resize the cropped image
        depth_face = cv2.resize(depth_face, (224, 224))
        
        # Save the cropped depth image
        cv2.imwrite(depth_save_path, depth_face)

    # Crop thermal image using scaled coordinates and margin
    thermal_frame = cv2.imread(thermal_path)
    if thermal_frame is not None:
        thermal_h, thermal_w = thermal_frame.shape[:2]
        
        # Scale coordinates and margin based on thermal image dimensions
        thermal_x = int(x * (thermal_w / texture_w))
        thermal_y = int(y * (thermal_h / texture_h))
        thermal_margin = int(margin * (thermal_w / texture_w))  # Assuming aspect ratio is preserved
        
        # Ensure the coordinates are within the bounds of the thermal image
        x_min = max(0, thermal_x - thermal_margin)
        x_max = min(thermal_w, thermal_x + thermal_margin)
        y_min = max(0, thermal_y - thermal_margin)
        y_max = min(thermal_h, thermal_y + thermal_margin)
        
        # Crop the thermal image
        thermal_face = thermal_frame[y_min:y_max, x_min:x_max]
        
        # Resize the cropped image
        thermal_face = cv2.resize(thermal_face, (224, 224))
        
        # Save the cropped thermal image
        cv2.imwrite(thermal_save_path, thermal_face)

def find_valid_path(base_dir, alternatives):
    """Find a valid path from a list of alternatives."""
    for alt in alternatives:
        path = os.path.join(base_dir, alt)
        if os.path.exists(path):
            return path
    return None

def process_images_multi_modal(texture_base_path: str, depth_base_path: str, thermal_base_path: str, save_dir: str, max_workers=None) -> None:
    os.makedirs(save_dir, exist_ok=True)
    
    # Define alternative folder names
    texture_alternatives = ["Texture_crop", "TextureCrop", "texture_crop", "texturecrop"]
    depth_alternatives = ["Depth_crop", "DepthCrop", "depth_crop", "depthcrop"]
    thermal_alternatives = ["Thermal_crop", "ThermalCrop", "thermal_crop", "thermalcrop"]
    
    # Check if the provided paths exist, if not, try alternatives
    if not os.path.exists(texture_base_path):
        texture_base_path = find_valid_path(os.path.dirname(texture_base_path), texture_alternatives)
        if texture_base_path:
            print(f"Using alternative texture path: {texture_base_path}")
        else:
            print("No valid texture path found.")
            return
    
    if not os.path.exists(depth_base_path):
        depth_base_path = find_valid_path(os.path.dirname(depth_base_path), depth_alternatives)
        if depth_base_path:
            print(f"Using alternative depth path: {depth_base_path}")
        else:
            print("No valid depth path found.")
            return
    
    if not os.path.exists(thermal_base_path):
        thermal_base_path = find_valid_path(os.path.dirname(thermal_base_path), thermal_alternatives)
        if thermal_base_path:
            print(f"Using alternative thermal path: {thermal_base_path}")
        else:
            print("No valid thermal path found.")
            return

    # Debug: Check if the base paths exist
    print(f"Checking base paths:")
    print(f"Texture base path: {texture_base_path}")
    print(f"Depth base path: {depth_base_path}")
    print(f"Thermal base path: {thermal_base_path}")

    # Use glob to find all texture images
    texture_files = glob.glob(f"{texture_base_path}/**/*.jpg", recursive=True)
    print(f"Found {len(texture_files)} texture images.")

    # Debug: Check a few texture files
    if texture_files:
        print("Sample texture files:")
        for i in range(min(5, len(texture_files))):
            print(f"  {texture_files[i]}")

    # Prepare a list of tuples containing paths for processing
    image_sets = []
    missing_depth = 0
    missing_thermal = 0
    
    for texture_path in texture_files:
        # Calculate the relative path for depth and thermal images
        relative_path = os.path.relpath(texture_path, texture_base_path)
        depth_path = os.path.join(depth_base_path, relative_path)
        thermal_path = os.path.join(thermal_base_path, relative_path)
        
        # Check if depth and thermal images exist
        depth_exists = os.path.exists(depth_path)
        thermal_exists = os.path.exists(thermal_path)
        
        if not depth_exists:
            missing_depth += 1
        if not thermal_exists:
            missing_thermal += 1
        
        # Append the set only if all modalities exist
        if depth_exists and thermal_exists:
            image_sets.append((texture_path, depth_path, thermal_path, save_dir))
    
    print(f"Missing depth images: {missing_depth}")
    print(f"Missing thermal images: {missing_thermal}")
    print(f"Processing {len(image_sets)} complete image sets.")

    # If no valid image sets were found, exit
    if not image_sets:
        print("No valid image sets found. Exiting.")
        return

    # Try with a smaller batch size or sequential processing if multiprocessing fails
    try:
        # Use ProcessPoolExecutor for concurrent processing with a progress bar
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Create a progress bar
            with tqdm(total=len(image_sets), desc="Processing images") as pbar:
                # Process images and update progress bar
                for _ in executor.map(process_single_image_set, image_sets):
                    pbar.update(1)
    except Exception as e:
        print(f"Error with multiprocessing: {e}")
        print("Falling back to sequential processing...")
        
        # Sequential processing as a fallback
        with tqdm(total=len(image_sets), desc="Processing images") as pbar:
            for image_set in image_sets:
                process_single_image_set(image_set)
                pbar.update(1)

# def process_images(image_path: str, output_path: str, max_workers: int = 8):
#     print("Processing images in face cropping")
#     print(f"Processing {image_path} in face cropping")
#     # Create output directory if it doesn't exist
#     Path(output_path).mkdir(parents=True, exist_ok=True)
#
#     # Use glob to find all .jpg files in subdirectories
#     files = glob.glob(f"{image_path}/**/*.jpg", recursive=True)
#     print(f"There are {len(files)} images to process.")
#
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = []
#
#         for file in tqdm(files, desc="Processing images: crop_face_img"):
#             # Create the corresponding save path
#             # Replace the image_path with output_path in the file path
#             save_path = file.replace(image_path, output_path)
#
#             # Create the directory for the save path if it doesn't exist
#             Path("/".join(save_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
#
#             # Submit the cropping task to the executor
#             futures.append(executor.submit(crop_face_img, file, save_path))
#
#         # Wait for all futures to complete
#         for future in tqdm(futures, desc="Waiting for crops to finish"):
#             future.result()

