# tests/test_vector_store.py

from models.vector_store import add_text, search_similar

def test_vector_store_add_and_search():
    sample_text = "This is a test image of a cat in the garden"
    add_text(sample_text)
    
    results = search_similar("cat")
    assert isinstance(results, list)
    assert any("cat" in r.lower() for r in results)
