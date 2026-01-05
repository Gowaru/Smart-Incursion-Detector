from ultralytics import YOLO
import cv2

try:
    model = YOLO("yolo11n.pt")
    print("Model loaded")
    
    # Create a dummy frame
    import numpy as np
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Try tracking
    print("Attempting tracking with local bytetrack.yaml...")
    results = model.track(frame, tracker="bytetrack.yaml", persist=True)
    print("Tracking success!")
    
except Exception as e:
    print(f"Error: {e}")
