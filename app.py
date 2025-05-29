
import nest_asyncio
nest_asyncio.apply()

import asyncio
import sys

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from PIL import Image

from models.blip_model import extract_caption
from models.vector_store import (
    init_milvus_collection,
    get_embedding,
    get_image_id,
    insert_to_milvus,
    search_similar
)
from models.gpt_model import generate_response_stream
from models.image_embedder import ImageEmbedder

image_embedder = ImageEmbedder()

st.set_page_config(page_title="Image to Text RAG con Milvus", layout="wide")

collection = init_milvus_collection()

st.title("Image to Text RAG con Milvus")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_embedding = image_embedder.encode_image(image)

    # ⛔️ STOP se non è valido
    if image_embedding is None:
        st.error("❌ Errore: impossibile generare l'embedding dell'immagine. Verifica il modello.")
        st.stop()

    # Debug opzionale
    # st.write("image_embedding:", image_embedding[:5])
    st.write("🔍 image_embedding type:", type(image_embedding))
    st.write("🔍 image_embedding preview:", image_embedding[:5])


    uploaded_file.seek(0)
    image_bytes = uploaded_file.read()
    image_id = get_image_id(image_bytes)


    with st.spinner("Estrazione descrizione..."):
        caption = extract_caption(image)

    st.markdown("### 📷 Descrizione Base")
    st.markdown(f"> {caption}")

    # Embedding + insert
    embedding = get_embedding(caption)
    insert_to_milvus(collection, embedding, image_embedding, caption, image_id)

    # Cerca simili
    similar_texts = search_similar(collection, embedding, exclude_image_id=image_id)

    st.markdown("### 🧠 Descrizioni simili trovate")
    for i, (txt, _) in enumerate(similar_texts):
        st.markdown(f"{i+1}. {txt}")

    # Caption migliorata (con GPT)
    if similar_texts:
        with st.spinner("📚 Sto migliorando la descrizione usando il contesto..."):
            context = "\n".join([t[0] for t in similar_texts])
            prompt = f"""Questa è una descrizione automatica dell'immagine: "{caption}"

            Descrizioni simili da immagini precedenti:
            {context}

            Analizza attentamente il contenuto visivo descritto.
            - Se ci sono più oggetti, elencali con quantità e disposizione.
            - Se è presente testo visibile o simboli, includili nella descrizione.
            - Se l'immagine è un collage o una griglia, descrivi ogni sezione.
            - Se alcuni oggetti sono parziali o sfocati, menzionalo.

            Fornisci una **descrizione migliorata**, dettagliata ma concisa:"""

            response_placeholder = st.empty()
            improved_caption = ""
            for token in generate_response_stream([context], prompt, lang="it"):
                improved_caption += token
                response_placeholder.markdown(f"### ✨ Descrizione migliorata\n{improved_caption}")

    # Domanda dell'utente
    combined_context = [caption] + [t[0] for t in similar_texts if t[0] != caption]
    question = st.text_input("Fai una domanda sull'immagine")

    if question:
        response_placeholder = st.empty()
        answer = ""
        for token in generate_response_stream(combined_context, question, lang="it"):
            answer += token
            response_placeholder.markdown(answer)
