from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import torch

class ImageEmbedder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_image(self, image: Image.Image) -> list[float]:
      try:
          inputs = self.processor(images=image, return_tensors="pt")
          inputs = {k: v.to(self.device) for k, v in inputs.items()}
          with torch.no_grad():
              outputs = self.model.get_image_features(**inputs)
          embedding = outputs[0] / outputs[0].norm(dim=-1, keepdim=True)
          return embedding.cpu().numpy().tolist()
      except Exception as e:
          print(f"Errore in encode_image: {e}")
          return None

