
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
import hashlib

connections.connect("default", host="localhost", port="19530")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="image_id", dtype=DataType.VARCHAR, max_length=1024)
]

schema = CollectionSchema(fields, description="Image text embeddings with image ID")
collection_name = "rag_image_texts"

if collection_name not in utility.list_collections():
    collection = Collection(collection_name, schema)
    collection.create_index(field_name="embedding", index_params={
        "index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}
    })
else:
    collection = Collection(collection_name)

collection.load()

def get_image_id(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

def exists_text_with_image(text, image_id):
    safe_text = text.replace('"', '\\"')
    expr = f'image_id == "{image_id}" && text == "{safe_text}"'
    results = collection.query(expr=expr, output_fields=["text"])
    return len(results) > 0

def add_text_with_image(text, image_id):
    if not exists_text_with_image(text, image_id):
        embedding = embedder.encode([text])[0].tolist()
        collection.insert([
            [embedding],  # Milvus si aspetta lista di vettori
            [text],
            [image_id]
        ])
        collection.flush()
        print(f"Added text for image_id={image_id}")
    else:
        print(f"Text already exists for image_id={image_id}")

def search_similar_captions(query_text, exclude_image_id=None, top_k=5, distance_threshold=0.6):
    query_embedding = embedder.encode([query_text])[0].tolist()

    expr = None
    if exclude_image_id:
        expr = f'image_id != "{exclude_image_id}"'

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        expr=expr,
        output_fields=["text", "image_id"]
    )

    print(f"Search returned {len(results[0])} results")
    filtered = []
    for hit in results[0]:
        print(f"Candidate: text='{hit.entity.get('text')}' distance={hit.distance:.4f}")
        if hit.distance < distance_threshold:
            filtered.append((hit.entity.get("text"), hit.entity.get("image_id")))
            print(f" --> Accepted (distance {hit.distance:.4f})")

    return filtered
