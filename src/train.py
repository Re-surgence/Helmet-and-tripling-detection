from ultralytics import YOLO
import os
import shutil
import random

# Define paths (adjust as needed)
base_dir = 'Dataset'  # Relative to repo root
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

def split_dataset(train_dir, val_dir, val_split=0.2):
    images_dir = os.path.join(train_dir, 'images')
    labels_dir = os.path.join(train_dir, 'labels')
    
    val_images_dir = os.path.join(val_dir, 'images')
    val_labels_dir = os.path.join(val_dir, 'labels')
    
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    
    num_val = int(len(images) * val_split)
    val_images = images[:num_val]
    
    for img in val_images:
        src_img = os.path.join(images_dir, img)
        dst_img = os.path.join(val_images_dir, img)
        shutil.move(src_img, dst_img)
        
        label = os.path.splitext(img)[0] + '.txt'
        src_label = os.path.join(labels_dir, label)
        dst_label = os.path.join(val_labels_dir, label)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)

# Check and split if necessary
if not os.path.exists(os.path.join(val_dir, 'images')) or len(os.listdir(os.path.join(val_dir, 'images'))) == 0:
    print("Validation set missing. Splitting train set...")
    split_dataset(train_dir, val_dir)

# Create data.yaml
yaml_content = """
train: Dataset/train/images
val: Dataset/val/images

nc: 3
names: ['Bike', 'Head-With-helmet', 'Head-Without-Helmet']
"""
with open('data/data.yaml', 'w') as f:
    f.write(yaml_content)

def train_model(epochs=50, batch=16, imgsz=640):
    """Train YOLOv8n for 50 epochs on helmet-tripling dataset."""
    model = YOLO('yolov8n.pt')  # Pretrained nano
    model.train(data='data/data.yaml', epochs=epochs, imgsz=imgsz, batch=batch)
    model.save('models/helmet_tripling_model.pt')
    print("Training complete! Model saved to models/helmet_tripling_model.pt")

def evaluate_model(model_path='models/helmet_tripling_model.pt'):
    """Evaluate trained model on validation set."""
    model = YOLO(model_path)
    results = model.val()
    print("Validation mAP@0.5:", results.box.map)
    return results

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    train_model(epochs=50, batch=16)  # Your settings
    evaluate_model()  # Auto-runs after training