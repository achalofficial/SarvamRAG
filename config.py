import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.getenv("GEN_MODEL", "phi3:latest")

EMBED_DIM = 768
TURBOVEC_BITS = 4
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
INDEX_DIR = DATA_DIR / "index"
INDEX_FILE = INDEX_DIR / "sarvam_rag.tvec"
METADATA_FILE = INDEX_DIR / "chunks_metadata.json"

EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
