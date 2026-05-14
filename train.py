import torch
from ultralytics import YOLO

def main():
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

    model.train(
        data="self-labled-dataset-yolo/data.yaml",
        epochs=1,
        imgsz=1280,
        device=compute_device,
        project="local_bee_models",
        name="yolo26_run_01",
    )

if __name__ == "__main__":
    main()
