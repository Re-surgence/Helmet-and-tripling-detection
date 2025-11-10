from ultralytics import YOLO
import cv2
from utils import get_tripling_detections

def process_video(model_path='', video_path='', output_path='', conf_threshold=0.75): # Add your paths here
    """Process video for helmet-tripling detection."""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_id = 0
    class_names = ['Bike', 'Head-With-helmet', 'Head-Without-Helmet']
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        results = model(frame, conf=conf_threshold)[0]
        
        if results.boxes is None:
            out.write(frame)
            continue
        
        tripling_bikes, bike_indices, boxes, classes, scores = get_tripling_detections(results, conf_threshold)
        
        if tripling_bikes:
            print(f"Frame {frame_id}: Tripling on {len(tripling_bikes)} bikes")
        
        # Draw detections
        for i, (cls, box, score) in enumerate(zip(classes, boxes, scores)):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if cls == 0 else (255, 0, 0) if cls == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_names[int(cls)]} {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Highlight tripling
        for bike_idx in tripling_bikes:
            bike_box = boxes[bike_indices[bike_idx]]
            x1, y1, x2, y2 = map(int, bike_box)
            cv2.putText(frame, "Tripling!", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Video complete. Output: {output_path}")

if __name__ == '__main__':
    process_video(video_path='', conf_threshold=0.5) # Add your video path here
