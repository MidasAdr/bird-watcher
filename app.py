import config

from src.detection.detection_state import DetectionState
from src.audio.audio_listener import AudioListener
from src.camera.camera_stream import CameraStream
from src.detection.bird_detector import BirdDetector
from src.detection.audio_species_identifier import SpeciesIdentifier


def main():

    state = DetectionState()

    species_identifier = SpeciesIdentifier(state)

    detector = BirdDetector()

    camera = CameraStream(detector, state)

    if config.ENABLE_AUDIO_DETECTION:

        audio_listener = AudioListener(species_identifier, state)
        audio_listener.start()

    camera.start()


if __name__ == "__main__":
    main()
