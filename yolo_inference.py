from ultralytics import YOLO

# Load model
model = YOLO("models/best.pt")

results = model.predict("input_videos/08fd33_4.mp4", save=True)

print(results[0])