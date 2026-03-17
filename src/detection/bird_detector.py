from ultralytics import YOLO
from src.detection.image_species_classifier import ImageSpeciesClassifier
import config


class BirdDetector:

    def __init__(self):

        self.model = YOLO(config.YOLO_MODEL)
        self.classifier = ImageSpeciesClassifier()

    def detect(self, frame):

        results = self.model(frame, verbose=False)

        birds = []

        for r in results:

            for box in r.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                label = self.model.names[cls]

                if label == "bird" and conf > config.BOX_CONFIDENCE:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    birds.append((x1, y1, x2, y2, conf))

        return birds

    def identify_species(self, crop):

        if not config.ENABLE_IMAGE_DETECTION:
            return None
        try:
            return self.classifier.predict(crop)
        except Exception as e:
            print("Image classifier error:", e)
            return None
