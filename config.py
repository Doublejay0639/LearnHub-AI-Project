import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")


def resolve_path(env_var: str, default_path: Path) -> str:
    raw = os.environ.get(env_var)
    if raw:
        path = Path(raw)
        return str(path if path.is_absolute() else (BASE_DIR / path).resolve())
    return str(default_path.resolve())

CHROMA_DB_PATH = resolve_path("CHROMA_DB_PATH", BASE_DIR / "chroma_db")
COURSE_PDFS_PATH = resolve_path("COURSE_PDFS_PATH", BASE_DIR / "Course_PDFs")

# Ensure required directories exist for local and deployed environments
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(COURSE_PDFS_PATH, exist_ok=True)