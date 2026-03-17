import datetime
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

import config
from src.bird_data.bird_names import BIRD_NAMES


class SpeciesIdentifier:

    def __init__(self, state):
        self.state = state
        self.analyzer = Analyzer()

    def identify(self, wav_path):
        recording = Recording(
            self.analyzer,
            wav_path,
            lat=config.LAT,
            lon=config.LON,
            date=datetime.date.today()
        )

        recording.analyze()

        if not recording.detections:
            return None

        best = max(recording.detections, key=lambda x: x["confidence"])

        if best["confidence"] < config.SPECIES_CONFIDENCE:
            return None

        common = best["common_name"]

        scientific = None

        for k, v in BIRD_NAMES.items():
            if v["common"].lower() == common.lower():
                scientific = v["scientific"]
                break

        if not scientific:
            scientific = common

        return {
            "common": common,
            "scientific": scientific,
            "confidence": best["confidence"]
        }