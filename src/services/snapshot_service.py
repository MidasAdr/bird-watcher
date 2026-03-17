import cv2
import datetime
import os
import config
from src.utils.file_utils import safe_filename


def save_snapshot(frame, box, species):

    x1, y1, x2, y2 = box

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return

    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{safe_filename(species)}_{timestamp}.jpg"

    path = os.path.join(config.SNAPSHOT_DIR, filename)

    cv2.imwrite(path, crop)