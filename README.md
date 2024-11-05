# CORAL (Computer Orchestrated Recognition of Aquatic Life)

C.O.R.A.L (or CORAL) is a project that aims to create a system that can identify fish across cameras set up in the ocean.

In order to do this, we will be finetuning a YoloV11 model on multiple aquatic life detection datasets. We will be releasing the model weights to the scientific community for further research.

## Installation

To install the project, clone the repository and then install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> [!NOTE]
> This only works on Linux and MacOS. For Windows, you use `venv/Scripts/activate` instead of `source venv/bin/activate`.
 
## Usage

This project has two main components, the data collection and the model training.

### Data Collection

For the data, the project uses the [missfish](https://github.com/DianZhang/missfish) dataset. The dataset is not included in this repository, but can be downloaded from the link provided.

To set up the data, include all `.mp4` files and the `.pik` file inside the `raw/` folder. Then run:

```bash
python src/transform.py missfish
```

This will transform the data into the correct format for training, which is:

```
data/
    train/
        images/
            000000.jpg
            000001.jpg
            ...
        labels/
            000000.txt
            000001.txt
            ...
    val/
        images/
            000000.jpg
            000001.jpg
            ...
        labels/
            000000.txt
            000001.txt
            ...
```

To make sure this has worked, run:

```bash
python src/transform.py view <path_to_image> <path_to_label>
```

### Model Training

To train the model, run:

```bash
python src/train.py
```

