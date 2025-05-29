import os
from glob import glob
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import torch

import numpy as np  
class ImageEmbedder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_image(self, images: list[Image.Image]) -> list[list[float]]:
        try:
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            print(f"Errore in encode_images: {e}")
            return [None] * len(images)


def process_amazon_folder(image_folder, image_embedder, collection, batch_size=8, img_size=160):
    image_paths = glob(os.path.join(image_folder, "*.jpg"))

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = []

        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((img_size, img_size))
                images.append(img)
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
                images.append(None)  # placeholder per mantenere ordine

        valid_images = [img for img in images if img is not None]
        embeddings = image_embedder.encode_image(valid_images)

        emb_idx = 0
        for idx, img in enumerate(images):
            if img is None:
                print(f"Skipping image at batch index {idx} due to loading error")
                continue

            image_emb = embeddings[emb_idx]
            emb_idx += 1

            if image_emb is None:
                print(f"Skipping image at batch index {idx}, embedding failed.")
                continue

            # Converti embedding in numpy array float32 (qui solo UNA volta)
            image_emb = np.array(image_emb, dtype=np.float32)

            img_path = batch_paths[idx]
            text = "Amazon product image"

            text_emb = get_embedding(text)
            text_emb = np.array(text_emb, dtype=np.float32)

            with open(img_path, "rb") as f:
                image_bytes = f.read()
            image_id = get_image_id(image_bytes)

            insert_to_milvus(collection, text_emb, image_emb, text, image_id)

        torch.cuda.empty_cache()
        print(f"Batch da {len(batch_paths)} immagini processato e memoria liberata.")
