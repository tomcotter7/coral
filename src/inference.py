import click
from ultralytics import YOLO
import os
from pathlib import Path
import uuid
import cv2

@click.command()
@click.argument("custom_model_file", type=click.Path(exists=True))
@click.argument("model_input", type=str)
@click.option("-o", "--output", type=click.Path(), help="Folder to save the output image.")
def inference(custom_model_file: str, model_input: str, output: str|None):
    
    model = YOLO(custom_model_file)
    results = model(model_input, stream=True)
    basename = os.path.basename(model_input)
    run_id = uuid.uuid4().hex[:6]
    
    file_ext = model_input.split(".")[-1]
    video_formats = ['mp4', 'avi', 'mov', 'mkv']
    is_video = file_ext in video_formats
    
    fps = None
    if is_video:
        cap = cv2.VideoCapture(model_input)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    print(f"Run ID: {run_id}")

    if output is None:
        
        os.makedirs(str(Path.cwd().parent / "outputs"), exist_ok=True)
        output = "outputs"

    folder = Path(__file__).cwd() / output

    for i, result in enumerate(results):
        if is_video and fps is not None:
            timestamp = i / fps
            time_str = '{:02d}:{:02d}:{:02d}'.format(
                int(timestamp // 3600),
                int((timestamp % 3600) // 60),
                int(timestamp % 60)
            )
            filename = str(folder / f"{basename}_{time_str}_{run_id}_result.jpg")
        else:
            filename = str(folder / f"{basename}_{i}_{run_id}_result.jpg")
        
        if len(result.boxes.cls) > 0:
            result.save(filename=filename)

@click.group()
def cli():
    pass

cli.add_command(inference)

if __name__ == "__main__":
    cli()
