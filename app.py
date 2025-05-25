import streamlit as st
from PIL import Image
from models.blip_model import extract_image_details
from models.gpt_model import generate_response_stream
from models.vector_store import add_text, search_similar

st.title("Image to Text Response with RAG Multilingue")

# Selezione lingua (italiano, inglese, francese come esempio)
lang = st.selectbox("Seleziona la lingua / Select language", ["it", "en", "fr"])

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Extracting image details..."):
        image_details = extract_image_details(image)

    # Salva il testo nel vector store
    add_text(image_details)

    st.write(f"**Image description:** {image_details}")

    # Cerca contesti simili basati sulla descrizione dell'immagine
    similar_contexts = search_similar(image_details)

    # Costruisce contesto combinato (nuova immagine + simili)
    combined_context = [image_details] + similar_contexts

    st.write("### Ask a question about the image / Fai una domanda sull'immagine")
    question = st.text_input("Your question / La tua domanda:")

    if question:
        st.write("**Answer / Risposta:**")

        response_placeholder = st.empty()
        streamed_answer = ""

        # Usa la descrizione + simili per generare la risposta, specificando la lingua
        for token in generate_response_stream(combined_context, question, lang=lang):
            streamed_answer += token
            response_placeholder.markdown(f"**{streamed_answer}**")
