import time
import cv2
import config

from src.services.snapshot_service import save_snapshot
from src.tracking.bird_tracker import BirdTracker
from src.bird_data.bird_names import BIRD_NAMES


class CameraStream:

    def __init__(self, detector, state):
        self.detector = detector
        self.state = state
        self.last_snapshot = 0
        self.tracker = BirdTracker()

    def start(self):

        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        print("Camera started")

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            self.state.remove_old_tracks()

            tracks = self.get_tracks(frame)
            self.state.bird_visible = bool(tracks)

            for track in tracks:
                self.process_bird(frame, track)

            self.draw_ui(frame)

            cv2.imshow("Birdwatch Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def get_tracks(self, frame):
        detections = self.detector.detect(frame)
        return self.tracker.update(detections)

    def process_bird(self, frame, track):

        x1, y1, x2, y2, bird_id = map(int, track)

        self.draw_box(frame, x1, y1, x2, y2)

        crop = frame[y1:y2, x1:x2]

        self.handle_image_detection(bird_id, crop, x1, y1, x2, y2)

        self.handle_audio_override(bird_id)

        self.draw_label(frame, bird_id, x1, y2)

        self.handle_snapshot(frame, bird_id, x1, y1, x2, y2)

    def draw_box(self, frame, x1, y1, x2, y2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def draw_label(self, frame, bird_id, x1, y2):

        if bird_id not in self.state.birds:
            return

        data = self.state.birds[bird_id]

        cv2.putText(
            frame,
            data["common"],
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            data["scientific"],
            (x1, y2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 255, 200),
            1
        )

    def handle_image_detection(self, bird_id, crop, x1, y1, x2, y2):

        if not config.ENABLE_IMAGE_DETECTION:
            return

        try:
            area = (x2 - x1) * (y2 - y1)

            if area < config.MIN_BIRD_SIZE ** 2:
                return

            species = self.detector.identify_species(crop)

            if not species:
                return

            species = species.replace(" ", "_")

            common, scientific = self.map_species(species)

            if bird_id not in self.state.birds:
                self.state.update_species(bird_id, {
                    "common": common,
                    "scientific": scientific
                })

        except Exception as e:
            print("Image detection error:", e)

    def handle_audio_override(self, bird_id):

        if not config.ENABLE_AUDIO_DETECTION:
            return

        if time.time() - self.state.audio_timestamp > 5:
            return

        if not self.state.audio_species:
            return

        if not self.state.bird_visible:
            return

        audio_common = self.state.audio_species["common"]
        audio_scientific = self.state.audio_species["scientific"]

        current = self.state.birds.get(bird_id)

        if not current or current["common"] != audio_common:
            self.state.update_species(bird_id, {
                "common": audio_common,
                "scientific": audio_scientific
            })

    def handle_snapshot(self, frame, bird_id, x1, y1, x2, y2):

        if bird_id not in self.state.birds:
            return

        if time.time() - self.last_snapshot < config.SAVE_COOLDOWN:
            return

        species_name = self.state.birds[bird_id]["common"]

        save_snapshot(frame, (x1, y1, x2, y2), species_name)

        self.last_snapshot = time.time()

    def map_species(self, species):

        if species in BIRD_NAMES:
            info = BIRD_NAMES[species]
            return info["common"], info["scientific"]

        scientific = species.replace("_", " ")
        return scientific, scientific

    def draw_ui(self, frame):

        mode = "Audio" if config.ENABLE_AUDIO_DETECTION else "Image"

        if config.ENABLE_IMAGE_DETECTION and config.ENABLE_AUDIO_DETECTION:
            mode = "Audio and Image"

        cv2.putText(
            frame,
            f"Detection mode: {mode}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
