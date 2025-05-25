# tests/test_gpt_model.py

from models.gpt_model import generate_response_stream

def test_generate_response_stream():
    context = ["A test image showing a sunny beach"]
    question = "C'è il mare?"
    
    stream = generate_response_stream(context, question)
    output = "".join(list(stream))  # raccoglie tutto
    assert isinstance(output, str)
    assert len(output) > 0
