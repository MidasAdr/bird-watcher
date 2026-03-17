import os
import datetime
import config


def log_species(species, confidence):

    file_exists = os.path.exists(config.LOG_FILE)

    with open(config.LOG_FILE, "a") as f:

        if not file_exists:
            f.write("timestamp,species,confidence\n")

        f.write(
            f"{datetime.datetime.now().isoformat()},{species},{confidence}\n"
        )