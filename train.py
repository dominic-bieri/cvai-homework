import os
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO
import torch

load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

project = rf.workspace("roboflow-100").project("bees-jt5in")
version = project.version(2)
dataset = version.download("yolov8", location="./dataset")

if torch.cuda.is_available():
    print("NVIDIA GPU (CUDA) detected!")
    compute_device = 'cuda:0'
elif torch.backends.mps.is_available():
    print("Apple Silicon GPU (MPS) detected!")
    compute_device = 'mps'
else:
    print("No compatible GPU found, defaulting to CPU.")
    compute_device = 'cpu'

model = YOLO('yolo26n.pt')

print(f"Starting YOLO local training on {compute_device}...")

# TODO replace with self-labled dataset
dataset_yaml_path = os.path.join(dataset.location, 'data.yaml')

results = model.train(
    data=dataset_yaml_path,
    epochs=150,
    imgsz=1280,
    batch=16,
    workers=8,
    cache='disk',
    device=compute_device,
    project="local_bee_models",
    name="yolo26_run_01",
    # small object improvements
    scale=0.3,    # limit random scaling so small bees stay small during augmentation
    mosaic=1.0,   # mosaic augmentation at full strength
)