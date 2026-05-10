import chromadb
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import shutil

from typing import Optional #new
from query import ask, generate_assessment
from ingest import ingest_file, delete_file
from config import COURSE_PDFS_PATH
from config import CHROMA_DB_PATH
from chromadb.utils import embedding_functions

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH) #new

embedding_fn  = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)
chroma_collection = chroma_client.get_or_create_collection(
    name="course_materials",
    embedding_function=embedding_fn,
)

app = FastAPI(title="Course AI Tutor API")

# ── Allow your website to call this API ───────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # We'll lock this down to your domain later
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


class AskRequest(BaseModel):
    question: str
    course:   Optional[str] = None
    module:   Optional[str] = None

#new
class AssessmentGenerationRequest(BaseModel):
    course: str = Field(..., description="Course name (must match folder name in Course_Materials/)")
    module: Optional[str] = Field(None,  description="Module name — leave BLANK if your files sit directly in the course folder")
    num_questions: int = Field(10, ge=1, le=50)


@app.post("/ask")
async def ask_question(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty.")

    result = ask(
        question=req.question,
        course=req.course,
        module=req.module,
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result

# ── Upload a course material ──────────────────────────────────────
# ── Supported file types ──────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".csv"}

# ── Upload a course material ──────────────────────────────────────
@app.post("/upload")
async def upload_material(
    file: UploadFile = File(...),
    course_name: str = Form(...),
    module_name: str = Form(None)
):
    if not course_name or not course_name.strip():
        raise HTTPException(status_code=400, detail="course_name is required.")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Save file to the course folder
    save_dir = Path(COURSE_PDFS_PATH) / course_name / (module_name or "")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / file.filename

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest into ChromaDB
    chunks_indexed = ingest_file(
        file_path=str(save_path),
        course_name=course_name,
        module_name=module_name or "",
    )

    if chunks_indexed == 0:
        raise HTTPException(
            status_code=422,
            detail="File was received but no text could be extracted from it."
        )

    return {
        "success":        True,
        "file_name":      file.filename,
        "chunks_indexed": chunks_indexed,
        "course":         course_name,
        "module":         module_name or None,
    }
    

#new
@app.post("/generate-assessment")
async def generate_assessment_endpoint(req: AssessmentGenerationRequest):
    """
    Generates MCQ assessment questions scoped to a course or module.
    Returns a draft JSON array for tutor review before saving.
    """
    result = generate_assessment(
        course=req.course,
        module=req.module,
        num_questions=req.num_questions
    )

    if "error" in result:
        status = 404 if "No materials found" in result["error"] or "empty" in result["error"] else 500
        raise HTTPException(status_code=status, detail=result["error"])
    return result


# ── Delete a course material ──────────────────────────────────────
@app.delete("/material/{file_name}")
def delete_material(file_name: str):
    result = delete_file(file_name)

    if not result["success"]:
        raise HTTPException(
            status_code=404,
            detail=result.get("message", "Delete failed.")
        )

    return result


#new

@app.get("/courses")
async def list_indexed_courses():
    total = chroma_collection.count()
    if total == 0:
        return {"message": "ChromaDB is empty. Run ingest.py first.", "courses": []}

    all_data  = chroma_collection.get(include=["metadatas"], limit=total)  # ← fix here
    metadatas = all_data.get("metadatas", [])

    course_map = {}
    for meta in metadatas:
        course = (meta.get("course") or "").strip()
        module = (meta.get("module") or "").strip()
        if not course:
            continue
        if course not in course_map:
            course_map[course] = set()
        if module:
            course_map[course].add(module)

    result = []
    for course_name, modules in course_map.items():
        result.append({
            "course": course_name,
            "modules": sorted(list(modules)) if modules else [],
            "note": "No module subfolders — files are directly in the course folder" if not modules else ""
        })

    return {
        "total_chunks": total,
        "courses": sorted(result, key=lambda x: x["course"])
    }


@app.get("/debug-meta")
async def debug_meta():
    total = chroma_collection.count()
    all_data = chroma_collection.get(include=["metadatas"], limit=total)
    metadatas = all_data.get("metadatas", [])
    # Return the last 20 chunks' metadata so we can see what keys are stored
    return {"sample": metadatas[-20:]}