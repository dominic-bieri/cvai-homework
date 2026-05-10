import torch
from ultralytics import YOLO

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

results = model.train(
    data="self-labled-dataset-yolo/data.yaml",
    epochs=150,
    imgsz=1280,
    rect=True,        # preserves 16:9 ratio instead of squashing to square
    device=compute_device,
    project="local_bee_models",
    name="yolo26_run_01",
    # small object improvements
    scale=0.3,    # limit random scaling so small bees stay small during augmentation
    mosaic=1.0,   # mosaic augmentation at full strength
)
