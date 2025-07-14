from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path='best.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
        return boxes
