# engine/train_detector.py
import os
import argparse
from pathlib import Path
from ultralytics import YOLO

ENGINE_DIR = Path(__file__).resolve().parent
DATASET_DIR = ENGINE_DIR / "datasets" / "table_tennis"
DEFAULT_YAML = DATASET_DIR / "data.yaml"


def train_table_tennis_model(
    data_yaml: Path = DEFAULT_YAML,
    base_model: str = "yolov8s.pt",  # Small model offers great accuracy/speed balance
    epochs: int = 50,
    imgsz: int = 1280,              # High resolution (1280px) is vital for small balls
    batch: int = 8,
    device: str = "mps",            # Use Apple Silicon GPU (MPS)
    project_name: str = "tt_detector",
):
    """
    Fine-tunes a YOLOv8 model for table tennis ball & table detection.
    """
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        print(f"\n[ERROR] Dataset configuration file not found at: {data_yaml}")
        print("Please download a YOLOv8-formatted table tennis dataset and place it in 'engine/datasets/table_tennis/'.")
        print("Expected structure:")
        print("  engine/datasets/table_tennis/")
        print("    ├── data.yaml")
        print("    ├── train/ (images & labels)")
        print("    └── val/   (images & labels)")
        return None

    print(f"\n=======================================================")
    print(f"Starting Table Tennis YOLOv8 Fine-Tuning on Apple Silicon ({device})")
    print(f"Base Model: {base_model} | Epochs: {epochs} | ImgSize: {imgsz}px | Batch: {batch}")
    print(f"Dataset YAML: {data_yaml}")
    print(f"=======================================================\n")

    # 1. Load pretrained base model
    model = YOLO(base_model)

    # 2. Train on table tennis dataset
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(ENGINE_DIR / "runs" / project_name),
        name="train",
        patience=15,          # Early stopping if no improvement for 15 epochs
        save=True,
        save_period=10,
        augment=True,         # Auto-apply data augmentations (flips, scaling, blur)
        plots=True,
        verbose=True
    )

    # 3. Path to best saved model weights
    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    target_weights = ENGINE_DIR / "tt_yolov8_best.pt"

    if best_weights.exists():
        import shutil
        shutil.copy2(best_weights, target_weights)
        print(f"\n[TRAINING COMPLETE] Best model saved to: {target_weights}")
        print("You can now pass 'tt_yolov8_best.pt' directly to MultiModelTracker!")

    return target_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Custom YOLOv8 Table Tennis Model")
    parser.add_argument("--data", type=str, default=str(DEFAULT_YAML), help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt", choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image resolution for training (1280 recommended)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (8 recommended for 16GB Mac)")
    parser.add_argument("--device", type=str, default="mps", help="Device: 'mps' for Mac GPU, 'cpu', or '0' for CUDA")

    args = parser.parse_args()

    train_table_tennis_model(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )
