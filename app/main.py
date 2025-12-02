import os
import uuid
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from supabase import create_client, Client
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ============ CONFIG ===============

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not (OPENAI_API_KEY and GROQ_API_KEY and SUPABASE_URL and SUPABASE_SERVICE_ROLE):
    raise RuntimeError("Faltan variables en el .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE)

# Parámetros simples de RAG
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CHUNKS = 200
EMBEDDING_DIM = 1536


# ========= APP FASTAPI =============

app = FastAPI(
    title="RAG UCB Commerce",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ucb-e-commerce.vercel.app", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========= UTILIDADES ===============

def chunk_text(text: str) -> List[str]:
    """
    Divide el texto en chunks con solapamiento, evitando duplicados
    y bucles infinitos.
    """
    text = text.strip()
    n = len(text)
    if n == 0:
        return []

    # Caso simple: texto corto → un solo chunk
    if n <= CHUNK_SIZE:
        return [text]

    chunks: List[str] = []
    start = 0
    count = 0

    while start < n and count < MAX_CHUNKS:
        end = min(start + CHUNK_SIZE, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            count += 1

        if end == n:
            # Llegamos al final del texto
            break

        new_start = end - CHUNK_OVERLAP

        # Asegurarnos de que avanzamos y no entramos en bucle
        if new_start <= start:
            break

        start = new_start

    # Opcional: quitar duplicados exactos por seguridad
    seen = set()
    unique_chunks: List[str] = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            unique_chunks.append(c)

    return unique_chunks


def embed_text(text: str) -> List[float]:
    """Obtiene embedding OpenAI text-embedding-3-small para un solo texto."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Obtiene embeddings en batch para una lista de textos."""
    if not texts:
        return []
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    # La API devuelve los embeddings en el mismo orden del input
    return [item.embedding for item in response.data]


# ========= ENDPOINT: subir documento ===============

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["text/plain", "application/octet-stream"]:
        raise HTTPException(400, "Solo archivos .txt")

    content = await file.read()
    try:
        text = content.decode("utf-8")
    except Exception:
        text = content.decode("latin1")

    if len(content) > 2_000_000:
        raise HTTPException(413, "Archivo demasiado grande (máx 2 MB).")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "El archivo no contiene texto útil.")

    source_id = str(uuid.uuid4())

    # Una sola llamada a OpenAI para todos los chunks
    embeddings = embed_texts(chunks)

    rows = []
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        rows.append({
            "source_id": source_id,
            "chunk_index": idx,
            "text": chunk,
            "embedding": emb
        })

    if rows:
        supabase.table("rag_ucbcommerce_chunks").insert(rows).execute()

    return {"source_id": source_id, "chunks_stored": len(rows)}


# ========= ENDPOINT: chat RAG =======================

@app.post("/chat")
async def chat(payload: dict):
    question = payload.get("question")
    top_k = payload.get("top_k", 5)
    source_id = payload.get("source_id")  # opcional

    if not question:
        raise HTTPException(400, "Falta 'question'")

    q_emb = embed_text(question)

    params = {
        "query_embedding": q_emb,
        "match_count": top_k,
        "filter_source": source_id
    }

    # RPC específica para esta tabla
    rpc = supabase.rpc("match_rag_ucbcommerce_chunks", params).execute()
    matches = rpc.data or []

    context = "\n\n".join([f"- {m['text']}" for m in matches])

    prompt = f"""
Eres un asistente experto del sistema de soporte de la UCB Commerce.
Tu objetivo es ayudar a los usuarios con información sobre la universidad y los productos disponibles en la tienda.

Usa exclusivamente el contexto proporcionado para responder.
El contexto puede contener información institucional y fichas de productos (con precio, stock, categoría, etc.).

Reglas:
1. Si te preguntan por disponibilidad o stock de un producto, revisa el contexto. Si el stock es 0, indica que está agotado.
2. Si te preguntan precios, da el precio exacto del contexto.
3. Si la información no está en el contexto, responde: "No tengo esa información".
4. Sé amable y conciso.

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""

    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = completion.choices[0].message.content

    return {
        "answer": answer,
        "chunks_used": matches
    }


# ========= ROOT ===============

@app.get("/")
def root():
    return {"status": "ok", "msg": "RAG UCB Commerce listo"}
