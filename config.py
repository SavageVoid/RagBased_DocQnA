import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " "]
MERGE_THRESHOLD = 0.5

TOP_K_RESULTS = 4
HYBRID_RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_documents"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

APP_TITLE = "RAG Document Q&A"
APP_ICON = "📄"
PAGE_LAYOUT = "wide"

ENABLE_MEMORY = True
MEMORY_MAX_TURNS = 5

ENABLE_QUERY_TRANSFORMATION = True

MAX_SELF_RAG_ITERATIONS = 2