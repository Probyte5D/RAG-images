import os
from PIL import Image
from models.blip_model import extract_image_details
from models.vector_store import add_text

DATA_DIR = "data"

def preload_images():
    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(DATA_DIR, filename)
            try:
                image = Image.open(image_path)
                caption = extract_image_details(image)
                print(f"[✓] {filename} → {caption}")
                add_text(caption)
            except Exception as e:
                print(f"[X] Errore con {filename}: {e}")

if __name__ == "__main__":
    preload_images()
