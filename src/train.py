from ultralytics import YOLO

def start_training():
    # Load the pre-trained lightweight YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Train the model on your unzipped data
    model.train(
        data="data.yaml",     # Path to your clean data.yaml file
        epochs=20,            # 20 iterations through the dataset
        imgsz=640,            # Standard image resolution for YOLO
        device="cpu"          # Runs on CPU
    )

if __name__ == "__main__":
    start_training()