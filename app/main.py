from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import logging

from app.scanner import WatchScanner
from app.models import ScanResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Lifespan: load scanner once at startup ---
scanner_instance: WatchScanner = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scanner_instance
    logger.info("Loading WatchScanner...")
    scanner_instance = WatchScanner()
    logger.info("WatchScanner ready.")
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title="Watch Scanner API",
    description="Upload a watch image → get identification + projected price",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_BYTES = 6_000_000  # 6 MB


@app.post("/scan", response_model=ScanResponse, summary="Scan a watch image")
async def scan_watch(file: UploadFile = File(...)):
    """
    Upload a watch image (JPEG / PNG / WebP, max 6 MB).
    Returns: identification details + projected price range from 28k watch records.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Use JPEG/PNG/WebP.")

    image_bytes = await file.read()

    if len(image_bytes) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Image too large. Max 6 MB.")

    if len(image_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Image too small or corrupted.")

    logger.info(f"Scanning image: {file.filename} ({len(image_bytes)/1024:.1f} KB)")

    result = await asyncio.to_thread(scanner_instance.scan, image_bytes)
    return result


@app.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    count = 0
    if scanner_instance and scanner_instance.vectorstore:
        try:
            count = scanner_instance.vectorstore._collection.count()
        except Exception:
            pass
    return HealthResponse(status="ok", watches_indexed=count)