import cv2
import numpy as np
from collections import defaultdict

def compute_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def get_tripling_detections(results, conf_threshold=0.75):
    """Extract and group heads by bikes, return tripling bikes."""
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    
    bike_detections = defaultdict(list)
    bike_boxes = []
    bike_indices = []
    
    for i, (cls, box, score) in enumerate(zip(classes, boxes, scores)):
        if cls == 0:  # Bike
            bike_boxes.append(box)
            bike_indices.append(i)
        elif cls in [1, 2]:  # Heads
            for j, bike_box in enumerate(bike_boxes):
                iou = compute_iou(box, bike_box)
                if iou > 0.1:
                    bike_detections[j].append(cls)
                    break
    
    tripling_bikes = [j for j, heads in bike_detections.items() if len(heads) > 2]
    return tripling_bikes, bike_indices, boxes, classes, scores