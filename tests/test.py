from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
text = "ישראל שחררה את האסירים אתמול מעזה."
embedding = model.encode(text)
print(f"Embedding shape: {embedding.shape}")