# engine/datasets/download_dataset.py
import os
import argparse
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent / "table_tennis"

def download_roboflow_dataset(api_key: str, workspace: str = "roboflow-universe", project: str = "table-tennis-ball", version: int = 1):
    """
    Downloads a YOLOv8-formatted table tennis dataset from Roboflow Universe.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Roboflow package not found. Installing 'roboflow'...")
        import subprocess
        subprocess.run(["pip", "install", "roboflow"], check=True)
        from roboflow import Roboflow

    print(f"\nDownloading '{project}' (v{version}) to {DATASETS_DIR}...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8", location=str(DATASETS_DIR))
    print(f"\nDataset ready! Configuration file at: {dataset.location}/data.yaml")
    return dataset.location

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Table Tennis Dataset from Roboflow")
    parser.add_argument("--api-key", type=str, required=True, help="Your Roboflow API Key (Free from roboflow.com)")
    parser.add_argument("--project", type=str, default="table-tennis-ball", help="Project name on Roboflow")
    parser.add_argument("--version", type=int, default=1, help="Dataset version")
    args = parser.parse_args()

    download_roboflow_dataset(args.api_key, project=args.project, version=args.version)
