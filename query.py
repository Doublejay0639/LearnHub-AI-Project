import json #new
import chromadb #new
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from typing import Optional #new
from ingest import ingest_file
from groq import Groq
# from ingest import index
from config import GROQ_API_KEY
from config import CHROMA_DB_PATH #new
from tavily import TavilyClient   # pip install tavily-python
from chromadb.utils import embedding_functions


tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH) #new

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)
chroma_collection = chroma_client.get_or_create_collection(
    name="course_materials",        # ← matches ingest.py and api.py
    embedding_function=embedding_fn,
)

groq_client = Groq(api_key=GROQ_API_KEY)


def ask(question: str, course: str = None, module: str = None) -> dict:

    # ── 1. Pull relevant chunks from ChromaDB ────────────────────────────────
    course_context    = ""
    course_references = []
    had_course_match  = False

    if chroma_collection.count() > 0:
        where_filter = None
        if course and course.strip():
            where_filter = {"course": {"$eq": course.strip()}}
            if module and module.strip():
                where_filter = {
                    "$and": [
                        {"course": {"$eq": course.strip()}},
                        {"module": {"$eq": module.strip()}},
                    ]
                }

        try:
            results = chroma_collection.query(
                query_texts=[question],
                n_results=min(20, chroma_collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            raw_docs      = results.get("documents", [[]])[0]
            raw_metadatas = results.get("metadatas",  [[]])[0]
            raw_distances = results.get("distances",  [[]])[0]

            print(f"[ask] course filter: {course!r}  module filter: {module!r}")
            print(f"[ask] {len(raw_docs)} chunks returned from ChromaDB")
            for i in range(len(raw_docs)):
                print(f"  dist={raw_distances[i]:.4f}  "
                      f"course={raw_metadatas[i].get('course')}  "
                      f"file={raw_metadatas[i].get('file_name')}")

            valid_chunks = [
                {"text": raw_docs[i], "meta": raw_metadatas[i]}
                for i in range(len(raw_docs))
                if raw_docs[i] and raw_docs[i].strip()
            ]

            if valid_chunks:
                had_course_match = True
                course_context = "\n\n---\n\n".join(
                    c["text"] for c in valid_chunks[:15]
                )
                seen_files = []
                for c in valid_chunks:
                    fname = c["meta"].get("file_name", "Unknown file")
                    if fname and fname not in seen_files:
                        seen_files.append(fname)
                course_references = seen_files

        except Exception as e:
            print(f"[ask] ChromaDB error: {e}")


    # ── 2. Tavily web search — always runs to supplement ─────────────────────
    web_references = []
    web_context    = ""

    try:
        search = tavily_client.search(
            query=question,
            search_depth="basic",
            max_results=3,
            include_answer=False,
        )
        for i, item in enumerate(search.get("results", []), start=1):
            title   = (item.get("title")   or "").strip()
            url     = (item.get("url")     or "").strip()
            content = (item.get("content") or "").strip()
            if title and url and content:
                web_references.append({
                    "index":   i,
                    "title":   title,
                    "url":     url,
                    "snippet": content[:300],
                })

        if web_references:
            web_context = "\n\n".join(
                f"[{r['index']}] {r['title']}\n{r['snippet']}"
                for r in web_references
            )

    except Exception as e:
        print(f"[ask] Tavily error: {e}")


    # ── 3. Build the prompt ───────────────────────────────────────────────────
    if had_course_match:
        system_prompt = """You are a university course tutor.
You have been given excerpts from the student's actual course materials, as well as some web sources for additional context.
Your primary source is ALWAYS the course material. Base your answer on it first and foremost.
You may use the web sources only to briefly add context or real-world examples that the course material does not cover.
When you use information from a web source, cite it inline like [1] or [2].
Do not cite the course material — it is your primary knowledge base, treat it as such.
Never say phrases like "based on the provided material", "according to the course material", or "the material mentions".
Never say "the course material does not cover this" — if the web sources fill the gap, use them silently.
Just answer naturally and directly as a knowledgeable tutor would."""
    else:
        system_prompt = """You are an intelligent tutor for a university learning platform.
No course material is available for this topic, so answer using the web sources provided and your general knowledge.
When you use information from a numbered web source, cite it inline like [1] or [2].
Answer clearly, accurately, and educationally.
Never mention that you were given context or documents."""

    # Build context block — course material always comes first
    if had_course_match and course_context and web_context:
        user_message = (
            f"COURSE MATERIAL:\n{course_context}\n\n"
            f"WEB SOURCES:\n{web_context}\n\n"
            f"STUDENT QUESTION:\n{question}"
        )
    elif had_course_match and course_context:
        user_message = (
            f"COURSE MATERIAL:\n{course_context}\n\n"
            f"STUDENT QUESTION:\n{question}"
        )
    elif web_context:
        user_message = (
            f"WEB SOURCES:\n{web_context}\n\n"
            f"STUDENT QUESTION:\n{question}"
        )
    else:
        user_message = question


    # ── 4. Call Groq ──────────────────────────────────────────────────────────
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
    except Exception as e:
        return {"error": f"Groq API call failed: {str(e)}"}

    answer = response.choices[0].message.content.strip()


    # ── 5. Return ─────────────────────────────────────────────────────────────
    return {
        "answer":            answer,
        "course_references": course_references,
        "web_references":    web_references,
        "context_used": {
            "had_course_match": had_course_match,
            "had_web_results":  len(web_references) > 0,
        },
    }


# Test it
if __name__ == "__main__":
    answer = ask("What is a server?")
    print(answer)


#new

def generate_assessment(course: str, module: Optional[str] = None, num_questions: int = 10) -> dict:

    # ── Step 1: Check DB is populated ───────────────────────────────────────
    total_docs = chroma_collection.count()
    if total_docs == 0:
        return {"error": "ChromaDB is empty. Please run ingest.py first."}

    # ── Step 2: Fetch ONLY metadata (no documents) to find the exact course name
    # This is lightweight — we're not pulling document text yet, just labels.
    all_meta = chroma_collection.get(include=["metadatas"])
    metadatas = all_meta.get("metadatas", [])

    # Build a map: lowercased_name → exact stored name
    course_name_map = {}
    for meta in metadatas:
        stored = (meta.get("course") or "").strip()
        if stored:
            course_name_map[stored.lower()] = stored

    if not course_name_map:
        return {"error": "No course metadata found in ChromaDB. Re-run ingest.py."}

    # ── Step 3: Case-insensitive match to find the exact stored course name ──
    search_key = course.lower().strip()
    exact_course_name = None

    if search_key in course_name_map:
        exact_course_name = course_name_map[search_key]       # exact match
    else:
        for key, name in course_name_map.items():
            if search_key in key or key in search_key:
                exact_course_name = name                       # partial match
                break

    if exact_course_name is None:
        available = list(course_name_map.values())
        return {
            "error": (
                f"No materials found for course '{course}'. "
                f"Courses currently in the database: {available}"
            )
        }

    # ── Step 4: Build the ChromaDB where filter ──────────────────────────────
    # Use the EXACT stored name now — so the filter works perfectly.
    if module and module.strip():
        where_filter = {
            "$and": [
                {"course": {"$eq": exact_course_name}},
                {"module": {"$eq": module.strip()}}
            ]
        }
        scope_label = f"module '{module.strip()}' of course '{exact_course_name}'"
    else:
        where_filter = {"course": {"$eq": exact_course_name}}
        scope_label = f"course '{exact_course_name}'"

    # ── Step 5: Semantic retrieval — small, targeted, token-safe ─────────────
    # We ask for chunks relevant to "assessment topics and key concepts",
    # NOT all documents. This is what keeps us well under the token limit.
    CHUNK_COUNT    = 8    # number of chunks to retrieve
    CHARS_PER_CHUNK = 450  # cap per chunk — keeps total context ~3,600 chars ≈ ~900 tokens

    try:
        results = chroma_collection.query(
            query_texts=["key concepts definitions principles topics covered in this material"],
            n_results=min(CHUNK_COUNT, total_docs),
            where=where_filter,
            include=["documents"]
        )
    except Exception as e:
        return {"error": f"ChromaDB query failed: {str(e)}"}

    raw_docs = results.get("documents", [[]])[0]
    context_chunks = [
        doc[:CHARS_PER_CHUNK] for doc in raw_docs if doc and doc.strip()
    ]

    if not context_chunks:
        # If module filter returned nothing, give a helpful message
        if module and module.strip():
            # List available modules for this course so the tutor knows what to use
            module_meta = chroma_collection.get(
                where={"course": {"$eq": exact_course_name}},
                include=["metadatas"]
            )
            modules_found = list({
                (m.get("module") or "").strip()
                for m in module_meta.get("metadatas", [])
                if (m.get("module") or "").strip()
            })
            hint = (
                f"Available modules for this course: {modules_found}"
                if modules_found
                else "No module subfolders found — files are stored directly in the course folder. Leave module blank."
            )
            return {"error": f"No materials found for module '{module}' in '{exact_course_name}'. {hint}"}
        return {"error": f"No documents found for course '{exact_course_name}' after filtering."}

    context = "\n\n---\n\n".join(context_chunks)

    # ── Step 6: Token budget check before calling Groq ───────────────────────
    # Rough estimate: 1 token ≈ 4 chars. We stay well under the 6000 TPM limit.
    # Context (~900 tokens) + prompt overhead (~350 tokens) + output (~150*N tokens)
    # For N=10: 900 + 350 + 1500 = 2750 tokens — comfortably under 6000.
    # For N=20: 900 + 350 + 3000 = 4250 tokens — still safe.

    # ── Step 7: Prompt — NOTE: no correct_answer hints in the instructions ───
    # The explanation is stored server-side only. The model just generates the
    # question, options, correct_answer key, and explanation for MongoDB storage.
    system_prompt = (
        "You are an expert educator. Generate high-quality multiple choice questions. "
        "Return ONLY a valid JSON array — no markdown, no preamble, no text outside the JSON."
    )

    user_prompt = f"""Generate exactly {num_questions} multiple choice questions for {scope_label}.

Use ONLY the course material below. Cover a variety of topics. Mix difficulty levels.

COURSE MATERIAL:
\"\"\"
{context}
\"\"\"

Return a JSON array in this exact format — nothing else:
[
  {{
    "question": "Question text?",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct_answer": "A",
    "explanation": "Brief reason why this answer is correct."
  }}
]"""

    # ── Step 8: Call Groq ────────────────────────────────────────────────────
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.65,
            max_tokens=2500,   # enough for up to ~15 well-formed questions
        )
    except Exception as e:
        return {"error": f"Groq API call failed: {str(e)}"}

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model adds them anyway
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # ── Step 9: Parse and validate ───────────────────────────────────────────
    try:
        questions = json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "error": f"Model returned malformed JSON: {str(e)}",
            "raw_snippet": raw[:300]
        }

    required = {"question", "options", "correct_answer", "explanation"}
    validated = [q for q in questions if required.issubset(q.keys())]

    return {
        "questions":  validated,
        "count":      len(validated),
        "scope":      scope_label,
        "course":     exact_course_name,
        "module":     module or None,
    }