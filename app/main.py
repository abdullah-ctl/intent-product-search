# app/main.py
from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/embed")
async def embed_text(req: Request):
    data = await req.json()
    text = data.get("text", "")
    embedding = model.encode(text).tolist()
    return {"embedding": embedding}
