import pickle
import os
import click
from tqdm import tqdm
import random
import logging

from preprocessing import convert_to_images

from pathlib import Path

@click.command()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def missfish(verbose: bool):
    """Main entrypoint for converting data from https://github.com/DianZhang/missfish to YOLO format."""

    if verbose:
        logging.basicConfig(level=logging.INFO)

    raw_dir = Path(__file__).parent.parent / "raw"
    train_dir = Path(__file__).parent.parent / "data" / "train"
    val_dir = Path(__file__).parent.parent / "data" / "val"

    for dir in [train_dir, val_dir]:
        dir.mkdir(exist_ok=True, parents=True)
        labels = dir / "labels"
        images = dir / "images"
        labels.mkdir(exist_ok=True, parents=True)
        images.mkdir(exist_ok=True, parents=True)

    with open(raw_dir /  "niap_2019_annotations_all.pik", "rb") as f:
        annotations = pickle.load(f)

    files = [str(path) for path in raw_dir.glob("*.mp4")]

    random.shuffle(files)


    for file in tqdm(files[:-1], total=len(files) - 1):
        filename = os.path.basename(file).split(".")[0]
        logging.info(f"Processing: {filename}")
        convert_to_images(file, "data/train", annotations[filename]['anntations'])
    
    convert_to_images(files[-1], "data/val", annotations[os.path.basename(files[-1]).split(".")[0]]['anntations'])

    with open(train_dir.parent / "data.yaml", "w") as f:
        f.write("train: ../data/train\n")
        f.write("val: ../data/val\n")
        f.write("nc: 1\n")
        f.write("names: ['animal']\n")

@click.command()
@click.argument("image")
@click.argument("labels")
def view(image: str, labels: str):
    """View the image and labels."""
    import cv2

    cv2_image = cv2.imread(image)
    image_height, image_width, _ = cv2_image.shape
    with open(labels, "r") as f:
        for line in f:
            line = line.strip().split()
            x_center, y_center, norm_w, norm_h = map(float, line[1:])

            norm_l = x_center - norm_w / 2
            norm_t = y_center - norm_h / 2

            x = int(norm_l * image_width)
            y = int(norm_t * image_height)
            w = int(norm_w * image_width)
            h = int(norm_h * image_height)
            
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(cv2_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("image", cv2_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

@click.group()
def cli():
    pass

cli.add_command(missfish)
cli.add_command(view)


if __name__ == "__main__":
    cli()

