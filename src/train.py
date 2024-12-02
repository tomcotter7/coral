from ultralytics import YOLO
from pathlib import Path
import click

@click.command()
def train():
    model = YOLO("yolo11n.pt")
    results = model.train(data=Path(__file__).parent.parent / "data" / "data.yaml", epochs=1)
    click.echo(results)

@click.command()
@click.argument("custom_model_file", type=click.Path(exists=True))
@click.option("-d", "--data", type=click.Path(exists=True), help="Path to the data.yaml file.")
def train_custom(custom_model_file: str, data: str|None):
    model = YOLO(custom_model_file)
    if data is None:
        data = str(Path(__file__).parent.parent / "data" / "data.yaml")
    results = model.train(data=data, epochs=1)

@click.group()
def cli():
    pass

cli.add_command(train)
cli.add_command(train_custom)


if __name__ == "__main__":
    cli()
