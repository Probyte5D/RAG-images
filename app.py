import streamlit as st
from PIL import Image
from models.blip_model import extract_image_details
from models.gpt_model import generate_response

st.title("Image to Text Response with RAG")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Extracting image details..."):
        image_details = extract_image_details(image)
    st.write(f"**Image description:** {image_details}")
    
    st.write("### Ask a question about the image")
    question = st.text_input("Your question:")
    
    if question:
        with st.spinner("Generating response..."):
            answer = generate_response([image_details], question)
        st.write(f"**Answer:** {answer}")
