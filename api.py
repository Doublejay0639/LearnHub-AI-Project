import chromadb
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from lime.lime_text import LimeTextExplainer
from query import ask
import numpy as np

from typing import Optional
from query import ask, generate_assessment
from ingest import ingest_all, ingest_from_url, delete_file
from config import CHROMA_DB_PATH, COURSE_PDFS_PATH
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

AUTO_INGEST = os.environ.get("AUTO_INGEST", "1").lower() not in {"0", "false", "no"}

# ── Allow your website to call this API ───────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # We'll lock this down to your domain later
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print(f"[startup] CHROMA_DB_PATH={CHROMA_DB_PATH}")
    print(f"[startup] COURSE_PDFS_PATH={COURSE_PDFS_PATH}")
    print(f"[startup] AUTO_INGEST={'enabled' if AUTO_INGEST else 'disabled'}")
    if AUTO_INGEST:
        try:
            print("[startup] Ingesting Course_PDFs...")
            before = chroma_collection.count()
            print(f"[startup] ChromaDB count before ingest: {before}")
            total_indexed = await asyncio.to_thread(ingest_all)
            after = chroma_collection.count()
            print(f"[startup] ChromaDB count after ingest: {after}")
            print(f"[startup] AUTO_INGEST complete. Ingest returned: {total_indexed}")
            # show a few sample metadata entries if any
            try:
                samples = chroma_collection.get(include=["metadatas"]).get("metadatas", [])[:10]
                print(f"[startup] Sample metadatas ({len(samples)}): {samples}")
            except Exception as e:
                print(f"[startup] Failed to read sample metadatas: {e}")
        except Exception as e:
            print(f"[startup] AUTO_INGEST failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[startup] AUTO_INGEST disabled")

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
    try:
        if not req.question or not req.question.strip():
            raise HTTPException(status_code=400, detail="question cannot be empty.")

        print(f"\n[ask] Question: {req.question[:80]}...")
        print(f"[ask] Course: {req.course or 'N/A'}, Module: {req.module or 'N/A'}")
        
        result = ask(
            question=req.question,
            course=req.course,
            module=req.module,
        )

        if "error" in result:
            print(f"[ask] ERROR: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])

        print(f"[ask] ✓ Answer generated")
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ask] EXCEPTION: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get answer from AI")

# ── Upload request model ──────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".csv"}

class UploadRequest(BaseModel):
    fileUrl:     str            # Cloudinary secure_url
    fileName:    str            # original file name e.g. "lecture1.pdf"
    course_name: str
    module_name: Optional[str] = None

# New model for /upload-by-url endpoint (backend sends snake_case)
class UploadByUrlRequest(BaseModel):
    file_url:    str
    file_name:   str
    course_name: str
    module_name: Optional[str] = None

# ── Upload a course material ──────────────────────────────────────
@app.post("/upload")
async def upload_material(req: UploadRequest):
    if not req.course_name or not req.course_name.strip():
        raise HTTPException(status_code=400, detail="course_name is required.")

    if not req.fileUrl or not req.fileUrl.strip():
        raise HTTPException(status_code=400, detail="fileUrl is required.")

    if not req.fileName or not req.fileName.strip():
        raise HTTPException(status_code=400, detail="fileName is required.")

    ext = Path(req.fileName).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Download from Cloudinary, parse, chunk, embed, store in ChromaDB
    chunks_indexed = ingest_from_url(
        file_url=req.fileUrl,
        file_name=req.fileName,
        course_name=req.course_name.strip(),
        module_name=(req.module_name or "").strip(),
    )

    if chunks_indexed == 0:
        raise HTTPException(
            status_code=422,
            detail="File was downloaded but no text could be extracted from it."
        )

    return {
        "success":        True,
        "file_name":      req.fileName,
        "file_url":       req.fileUrl,
        "chunks_indexed": chunks_indexed,
        "course":         req.course_name.strip(),
        "module":         req.module_name or None,
    }


# ── Alias for /upload (called from backend as /upload-by-url) ──────────────────
@app.post("/upload-by-url")
async def upload_material_by_url(req: UploadByUrlRequest):
    """Handles file URL uploads from backend (snake_case format)"""
    try:
        if not req.course_name or not req.course_name.strip():
            raise HTTPException(status_code=400, detail="course_name is required.")

        if not req.file_url or not req.file_url.strip():
            raise HTTPException(status_code=400, detail="file_url is required.")

        if not req.file_name or not req.file_name.strip():
            raise HTTPException(status_code=400, detail="file_name is required.")

        print(f"\n[upload-by-url] Processing: {req.file_name}")
        print(f"[upload-by-url] Course: {req.course_name}, Module: {req.module_name or 'N/A'}")

        ext = Path(req.file_name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        # Download from Cloudinary, parse, chunk, embed, store in ChromaDB
        print(f"[upload-by-url] Calling ingest_from_url...")
        chunks_indexed = ingest_from_url(
            file_url=req.file_url,
            file_name=req.file_name,
            course_name=req.course_name.strip(),
            module_name=(req.module_name or "").strip(),
        )

        if chunks_indexed == 0:
            error_msg = "File was downloaded but no text could be extracted from it."
            print(f"[upload-by-url] ERROR: {error_msg}")
            raise HTTPException(status_code=422, detail=error_msg)

        print(f"[upload-by-url] ✓ Success: {chunks_indexed} chunks indexed")
        return {
            "success":        True,
            "file_name":      req.file_name,
            "file_url":       req.file_url,
            "chunks_indexed": chunks_indexed,
            "course":         req.course_name.strip(),
            "module":         req.module_name or None,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[upload-by-url] EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    

#new
@app.get("/courses")
def list_courses():
    """Return a list of distinct courses currently indexed in ChromaDB."""
    try:
        data = chroma_collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        courses = sorted({(m.get("course") or "").strip() for m in metadatas if (m.get("course") or "").strip()})
        return {"success": True, "courses": courses}
    except Exception as e:
        print(f"[courses] ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list indexed courses")


@app.post("/generate-assessment")
async def generate_assessment_endpoint(req: AssessmentGenerationRequest):
    """
    Generates MCQ assessment questions scoped to a course or module.
    Returns a draft JSON array for tutor review before saving.
    """
    try:
        print(f"\n[generate-assessment] Generating for course: {req.course}")
        print(f"[generate-assessment] Module: {req.module or 'N/A'}, Questions: {req.num_questions}")
        
        result = generate_assessment(
            course=req.course,
            module=req.module,
            num_questions=req.num_questions
        )

        if "error" in result:
            print(f"[generate-assessment] ERROR: {result['error']}")
            status = 404 if "No materials found" in result["error"] or "empty" in result["error"] else 500
            raise HTTPException(status_code=status, detail=result["error"])
        
        print(f"[generate-assessment] ✓ Generated {len(result.get('questions', []))} questions")
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"[generate-assessment] EXCEPTION: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/explain")
async def explain(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty.")

    # Step 1 — get the course chunks first (reuse your existing ask logic)
    result = ask(question=req.question, course=req.course, module=req.module)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    # Step 2 — extract the answer text to use as reference
    original_answer = result["answer"]

    # Step 3 — define a classifier that measures how much each sentence
    # in the context contributes to producing a similar answer
    explainer = LimeTextExplainer(class_names=["irrelevant", "relevant"])

    STOPWORDS = {"the", "a", "an", "is", "it", "in", "of", "and", "or", "to", "that", "this", "for", "with", "as", "are", "be", "was", "were", "by", "at", "on", "can"}

    def classifier_fn(texts):
        scores = []
        for text in texts:
            original_words = set(original_answer.lower().split()) - STOPWORDS
            perturbed_words = set(text.lower().split()) - STOPWORDS
            overlap = len(original_words & perturbed_words) / max(len(original_words), 1)
            scores.append([1 - overlap, overlap])
        return np.array(scores)

    # Step 4 — use the answer itself as the text to explain
    try:
        exp = explainer.explain_instance(
            original_answer,
            classifier_fn,
            num_features=6,
            num_samples=100,
        )

        top_sentences = [
            {"sentence": word, "importance": round(float(score), 4)}
            for word, score in exp.as_list()
            if score > 0  # only return positively contributing terms
        ]

        # sort by importance descending
        top_sentences = sorted(top_sentences, key=lambda x: x["importance"], reverse=True)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LIME explanation failed: {str(e)}")

    return {
        "question": req.question,
        "answer": original_answer,
        "course_references": result.get("course_references", []),
        "explanation": {
            "description": "Words and phrases that most contributed to this answer",
            "top_features": top_sentences
        }
    }