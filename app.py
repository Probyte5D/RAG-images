import re
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import hashlib

from models.vector_store import add_text_with_image, search_similar_captions
from models.gpt_model import generate_response_stream

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def extract_image_details(image: Image.Image) -> str:
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def clean_html_tags(text):
    return re.sub(r'</?div[^>]*>', '', text).strip()

def get_image_id(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

st.title("Image to Text RAG con Milvus e Ollama")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    uploaded_file.seek(0)
    image_bytes = uploaded_file.read()
    image_id = get_image_id(image_bytes)

    with st.spinner("Estrazione descrizione..."):
        image_details = extract_image_details(image)

    st.markdown(f"**Descrizione immagine:** {image_details}")

    # Aggiungi descrizione+embedding solo se non presente
    add_text_with_image(image_details, image_id)

    # Cerca testi simili ESCLUDENDO la stessa immagine
    similar_texts = search_similar_captions(image_details, exclude_image_id=image_id)

    # Costruisci contesto unico: descrizione + testi simili filtrati da duplicati
    all_texts = [image_details] + [text for text, _ in similar_texts if text != image_details]
    combined_context = list(dict.fromkeys(all_texts))

    st.markdown("### Testi simili trovati:")
    for i, txt in enumerate(combined_context):
        st.markdown(f"{i}. {txt}")

    question = st.text_input("Fai una domanda sull'immagine")

    if question:
        response_placeholder = st.empty()
        streamed_answer = ""

        for token in generate_response_stream(combined_context, question, lang="it"):
            streamed_answer += token
            clean_answer = clean_html_tags(streamed_answer)
            response_placeholder.markdown(clean_answer)
