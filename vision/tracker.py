# engine/vision/tracker.py
import cv2
import torch
import numpy as np
from ultralytics import YOLO

class MultiModelTracker:
    def __init__(self, yolo_path: str, device: str = "cuda"):
        self.device = device
        # YOLO for Table/Court Bounds & Player Detections
        self.yolo = YOLO(yolo_path)
        
    def process_frame(self, frame: np.ndarray):
        """
        Runs object tracking and returns frame metadata
        """
        results = self.yolo(frame, verbose=False)[0]
        
        players = []
        table_polygon = None
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            coords = box.xyxy[0].cpu().numpy()
            
            if label == "person":
                players.append(coords)
                
        # Extract segmentation mask for table surface if available
        if results.masks is not None:
            masks = results.masks.xy
            if len(masks) > 0:
                table_polygon = masks[0]  # Polygon coordinates of court/table

        return {
            "players": players,
            "table_polygon": table_polygon
        }