import cv2
import os
import logging
from typing import Any

def convert_to_images(video_path: str, output_path: str, annotations: dict[int, list[dict[str, Any]]]):
    """Convert a video to set of images, and then save the respective image and its labels in the output path in YOLO format.

    This assumes that the annotations are of the form:
        - key: frame number
        - value: list of annotations for that frame, where each annotation is a dictionary with the following keys
            - boundingBox
    
    Since we also are creating a single class dataset, the actual tags are not used.

    Args:
        video_path (str): The path to the video file.
        output_path (str): The path to the output directory.
        annotations (dict[int, list[dict[str, Any]]]): The annotations for the video.

    Returns:
        None - The function saves the images and labels in the output directory.
    """
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        if frame_number % 100 == 0:
            logging.info(f"Processing frame {frame_number}...")

        success, frame = cap.read()
        if not success:
            break

        if len(annotations.get(frame_number, [])) > 0:
            img_height, img_width, _ = frame.shape
            labels = []
            for annotation in annotations[frame_number]:
                bbox = annotation["boundingBox"]
                height, width, left, top = int(bbox["height"]), int(bbox["width"]), int(bbox["left"]), int(bbox["top"])
                norm_l, norm_t, norm_w, norm_h = left / img_width, top / img_height, width / img_width, height / img_height
            

                class_id = 0

                x_center = norm_l + norm_w / 2
                y_center = norm_t + norm_h / 2

                labels.append(f"{class_id} {x_center} {y_center} {norm_w} {norm_h}")

            with open(f"{output_path}/labels/{os.path.basename(video_path)}_frame_{frame_number}.txt", "w") as f:
                f.write("\n".join(labels))

            cv2.imwrite(f"{output_path}/images/{os.path.basename(video_path)}_frame_{frame_number}.jpg", frame)

        frame_number += 1


