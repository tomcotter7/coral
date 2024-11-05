from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolo11n.pt")
results = model.train(data=Path(__file__).parent.parent / "data" / "data.yaml", epochs=1)
