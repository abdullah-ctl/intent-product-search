# app/main.py
from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("/models/fine-tuned")

@app.post("/embed")
async def embed_text(req: Request):
    data = await req.json()
    text = data.get("text", "")
    embedding = model.encode(text).tolist()
    return {"embedding": embedding}


@app.get("/")
async def root():
    return {"message": "Welcome to the Inter Based Product Search API!"}