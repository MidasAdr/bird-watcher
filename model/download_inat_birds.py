import requests
import os
from tqdm import tqdm
from PIL import Image
from io import BytesIO

species_list = [
    "Erithacus rubecula",
    "Turdus merula",
    "Parus major",
    "Cyanistes caeruleus",
    "Pica pica",
    "Corvus cornix",
    "Passer domesticus",
    "Columba palumbus",
    "Larus argentatus",
    "Sturnus vulgaris",
    "Phasianus colchicus",
    "Passer montanus",
    "Troglodytes troglodytes",
    "Fringilla coelebs",
    "Corvus frugilegus",
    "Corvus monedula",
    "Garrulus glandarius",
    "Accipiter nisus"
]

DATASET_DIR = "dataset"

os.makedirs(DATASET_DIR, exist_ok=True)


def download_species(species, max_images=400):

    folder = os.path.join(DATASET_DIR, species.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)

    url = "https://api.inaturalist.org/v1/observations"

    params = {
        "taxon_name": species,
        "photos": "true",
        "per_page": 200
    }

    r = requests.get(url, params=params).json()

    images = []

    for obs in r["results"]:
        for photo in obs["photos"]:
            images.append(photo["url"].replace("square", "large"))

    images = images[:max_images]

    for i, img_url in enumerate(tqdm(images, desc=species)):

        try:

            img = requests.get(img_url).content
            img = Image.open(BytesIO(img)).convert("RGB")

            img.save(f"{folder}/{i}.jpg")

        except:
            pass


for species in species_list:
    download_species(species)