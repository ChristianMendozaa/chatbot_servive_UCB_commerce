from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.rag_service import process_upload, get_answer

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["text/plain", "application/octet-stream"]:
        raise HTTPException(400, "Solo archivos .txt")

    content = await file.read()
    if len(content) > 2_000_000:
        raise HTTPException(413, "Archivo demasiado grande (máx 2 MB).")

    try:
        result = process_upload(content)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error interno: {str(e)}")

@router.post("/chat")
async def chat(payload: dict):
    question = payload.get("question")
    top_k = payload.get("top_k", 5)
    source_id = payload.get("source_id")  # opcional

    if not question:
        raise HTTPException(400, "Falta 'question'")

    try:
        return get_answer(question, top_k, source_id)
    except Exception as e:
        raise HTTPException(500, f"Error interno: {str(e)}")
