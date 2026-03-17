import threading
import time
import tempfile
import sounddevice as sd
import soundfile as sf
import config


class AudioListener:

    def __init__(self, species_identifier, state):
        self.species_identifier = species_identifier
        self.state = state
        self.running = True

    def start(self):
        t = threading.Thread(target=self.loop, daemon=True)
        t.start()

    def record_audio(self):
        frames = int(config.RECORD_SECONDS * config.SAMPLE_RATE)

        audio = sd.rec(
            frames,
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        return audio.flatten()

    def loop(self):

        while self.running:

            if not config.ENABLE_AUDIO_DETECTION:
                time.sleep(1.0)
                continue

            try:
                audio = self.record_audio()

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, audio, config.SAMPLE_RATE)
                    wav_path = tmp.name

                result = self.species_identifier.identify(wav_path)

                if result:
                    self.state.update_audio_species(
                        result["common"],
                        result["scientific"],
                        result["confidence"]
                    )

            except Exception as e:
                print("Audio error:", e)

            time.sleep(0.2)