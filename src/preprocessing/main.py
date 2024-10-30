import pickle
import cv2
import os

from pathlib import Path

classes = {
    "fish": 0,
    "crab": 1
}

def convert_to_images(video_path: str, output_path: str, annotations: dict):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        if frame_number % 100 == 0:
            print(f"Processing frame {frame_number}...")

        success, frame = cap.read()
        # what is the width and height of the frame?
        if not success:
            break

        if len(annotations.get(frame_number, [])) > 0:
            img_height, img_width, _ = frame.shape
            labels = []
            for annotation in annotations[frame_number]:
                bbox = annotation["boundingBox"]
                height, width, left, top = int(bbox["height"]), int(bbox["width"]), int(bbox["left"]), int(bbox["top"])
                norm_l, norm_t, norm_w, norm_h = left / img_width, top / img_height, width / img_width, height / img_height
                
                if len(annotation["tags"]) > 1:
                    print(f"Warning: More than one tag in annotation {annotation}. Using the first one.")
                    print(f"Tags: {annotation['tags']}")

                class_id = classes[annotation["tags"][0].lower()]

                x_center = norm_l + norm_w / 2
                y_center = norm_t + norm_h / 2

                labels.append(f"{class_id} {x_center} {y_center} {norm_w} {norm_h}")

            with open(f"{output_path}/labels/{os.path.basename(video_path)}_frame_{frame_number}.txt", "w") as f:
                f.write("\n".join(labels))

            cv2.imwrite(f"{output_path}/images/{os.path.basename(video_path)}_frame_{frame_number}.jpg", frame)

        frame_number += 1


def main():
    raw_dir = Path(__file__).parent.parent.parent / "raw"
    with open(raw_dir /  "niap_2019_annotations_all.pik", "rb") as f:
        annotations = pickle.load(f)

    files = [str(path) for path in raw_dir.glob("*.mp4")]
    for file in files:
        filename = os.path.basename(file).split(".")[0]
        convert_to_images(file, "data/train", annotations[filename]['anntations'])

if __name__ == "__main__":
    main()
