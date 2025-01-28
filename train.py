from ultralytics import YOLO

# Load pre-trained YOLO model (YOLOv5s is a small model, use 'yolov5m.pt', 'yolov5l.pt' or 'yolov5x.pt' for larger models)
model = YOLO('yolov5s.pt')  # Make sure the model weights are either present or can be downloaded

# Training configuration
model.train(
    data='data/dataset.yaml',  # Path to dataset configuration file (yaml format)
    epochs=50,                 # Number of epochs (iterations over the dataset)
    imgsz=640,                 # Image size (resize all images to 640x640 during training)
    batch=16,                  # Batch size (number of images processed in parallel per step)
    name='kebab_finder',       # Custom name for this training run (used for saving results)
    device='cpu'                   # GPU usage: use `0` for the first available GPU. Set to `'cpu'` if you don't have a GPU.
)
