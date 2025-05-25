# tests/test_blip_model.py

from PIL import Image
from models.blip_model import extract_image_details

def test_extract_image_details():
    image = Image.new("RGB", (100, 100), color="red")  # dummy image
    result = extract_image_details(image)
    assert isinstance(result, str)
    assert len(result) > 0
