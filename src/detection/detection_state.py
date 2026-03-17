import time


class DetectionState:

    def __init__(self):
        self.birds = {}
        self.bird_visible = False

        self.audio_species = None
        self.audio_timestamp = 0.0
        self.audio_confidence = 0.0

    def update_species(self, bird_id, species_data):
        if bird_id not in self.birds:
            self.birds[bird_id] = {
                "common": species_data["common"],
                "scientific": species_data["scientific"],
                "first_seen": time.time(),
                "last_seen": time.time()
            }
        else:
            self.birds[bird_id]["common"] = species_data["common"]
            self.birds[bird_id]["scientific"] = species_data["scientific"]
            self.birds[bird_id]["last_seen"] = time.time()

    def update_audio_species(self, common, scientific, confidence):
        self.audio_species = {
            "common": common,
            "scientific": scientific
        }
        self.audio_confidence = confidence
        self.audio_timestamp = time.time()

    def remove_old_tracks(self, timeout=5):
        now = time.time()
        remove_ids = [bid for bid, d in self.birds.items() if now - d["last_seen"] > timeout]
        for bid in remove_ids:
            del self.birds[bid]
