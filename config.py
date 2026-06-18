import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY    = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY  = os.environ.get("TAVILY_API_KEY")
CHROMA_DB_PATH  = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
COURSE_PDFS_PATH = os.environ.get("COURSE_PDFS_PATH", "./Course_PDFs")

# Ensure required directories exist for local and deployed environments
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(COURSE_PDFS_PATH, exist_ok=True)