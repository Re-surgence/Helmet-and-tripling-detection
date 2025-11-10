from ultralytics import YOLO
import cv2
from utils import get_tripling_detections

def detect_and_check_tripling(model_path='models/helmet_tripling_model.pt', image_path='data/test_image.jpg', conf_threshold=0.75):
    """Run inference on image and detect tripling."""
    model = YOLO(model_path)
    results = model(image_path, conf=conf_threshold)[0]
    
    tripling_bikes, bike_indices, boxes, classes, scores = get_tripling_detections(results, conf_threshold)
    
    if tripling_bikes:
        print(f"Tripling detected on {len(tripling_bikes)} bikes!")
    
    # Draw results
    img = cv2.imread(image_path)
    class_names = ['Bike', 'Head-With-helmet', 'Head-Without-Helmet']
    for i, (cls, box, score) in enumerate(zip(classes, boxes, scores)):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if cls == 0 else (255, 0, 0) if cls == 1 else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[int(cls)]} {score:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Highlight tripling
    for bike_idx in tripling_bikes:
        bike_box = boxes[bike_indices[bike_idx]]  # Fixed indexing
        x1, y1, x2, y2 = map(int, bike_box)
        cv2.putText(img, "Tripling!", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    output_path = 'outputs/output_image.jpg'
    cv2.imwrite(output_path, img)
    print(f"Output saved to {output_path}")
    return tripling_bikes

if __name__ == '__main__':
    detect_and_check_tripling(image_path='data/test_image.jpg', conf_threshold=0.5)  # Lower for better recall