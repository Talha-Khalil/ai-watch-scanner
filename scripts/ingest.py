"""
ingest.py — Build the ChromaDB vectorstore from your watches CSV.

CSV columns expected:
name, price, brand, model, ref, mvmt, bracem, yop

Example:
    uv run python scripts/ingest.py --csv data/watches.csv --rebuild
"""

import argparse
import logging
import os
import sys
import time
import shutil

import pandas as pd
from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

load_dotenv()


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./vectorstore")

# Hugging Face sentence-transformers model (must match app/scanner.py)
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Batch ingestion size
BATCH_SIZE = 2000


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clean_value(val) -> str:
    """Normalize CSV value."""
    if pd.isna(val) or val is None:
        return "Unknown"

    s = str(val).strip()
    return s if s else "Unknown"


def build_document_text(row: pd.Series) -> str:
    """
    Build the text that will be embedded.

    Richer semantic text → better vector search.
    """

    name = clean_value(row.get("name"))
    brand = clean_value(row.get("brand"))
    model = clean_value(row.get("model"))
    ref = clean_value(row.get("ref"))
    mvmt = clean_value(row.get("mvmt"))
    bracem = clean_value(row.get("bracem"))
    yop = clean_value(row.get("yop"))
    price = clean_value(row.get("price"))

    return (
        f"Watch: {name}. "
        f"Brand: {brand}. "
        f"Model: {model}. "
        f"Reference: {ref}. "
        f"Movement: {mvmt}. "
        f"Bracelet: {bracem}. "
        f"Year: {yop}. "
        f"Price: {price}."
    )


# ---------------------------------------------------------------------
# CSV Loading
# ---------------------------------------------------------------------

def load_csv(csv_path: str) -> pd.DataFrame:

    logger.info(f"Loading CSV: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    df.columns = [c.strip().lower() for c in df.columns]

    required = {"name", "price", "brand", "model", "ref", "mvmt", "bracem", "yop"}

    missing = required - set(df.columns)

    if missing:
        logger.error(f"Missing required columns: {missing}")
        logger.error(f"Found columns: {list(df.columns)}")
        sys.exit(1)

    logger.info(f"Loaded {len(df):,} rows")

    # Drop useless rows
    before = len(df)
    df = df[~(df["name"].isna() & df["brand"].isna())]

    dropped = before - len(df)

    if dropped:
        logger.info(f"Dropped {dropped} empty rows")

    return df


# ---------------------------------------------------------------------
# Embeddings Loader
# ---------------------------------------------------------------------

def load_embeddings() -> SentenceTransformer:
    logger.info(f"Loading Hugging Face embeddings: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    return model


# ChromaDB collection name (must match app/scanner.py)
COLLECTION_NAME = "watches"


# ---------------------------------------------------------------------
# Vectorstore Builder
# ---------------------------------------------------------------------

def build_vectorstore(df: pd.DataFrame, rebuild: bool = False) -> None:
    if rebuild and os.path.exists(VECTORSTORE_DIR):
        logger.info(f"Deleting existing vectorstore at {VECTORSTORE_DIR}")
        shutil.rmtree(VECTORSTORE_DIR)

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    client = PersistentClient(path=VECTORSTORE_DIR)
    coll = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    embeddings_model = load_embeddings()
    total = len(df)
    batches = (total // BATCH_SIZE) + (1 if total % BATCH_SIZE else 0)
    logger.info(
        f"Ingesting {total:,} records in {batches} batches (batch size {BATCH_SIZE})"
    )
    start_time = time.time()

    for batch_num in range(batches):
        start = batch_num * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)
        batch = df.iloc[start:end]

        texts = [build_document_text(row) for _, row in batch.iterrows()]
        batch_embeddings = embeddings_model.encode(
            texts, normalize_embeddings=True
        ).tolist()

        ids = [str(start + i) for i in range(len(batch))]
        metadatas = []
        for _, row in batch.iterrows():
            metadatas.append({
                "name": clean_value(row.get("name")),
                "price": clean_value(row.get("price")),
                "brand": clean_value(row.get("brand")),
                "model": clean_value(row.get("model")),
                "ref": clean_value(row.get("ref")),
                "mvmt": clean_value(row.get("mvmt")),
                "bracem": clean_value(row.get("bracem")),
                "yop": clean_value(row.get("yop")),
            })

        coll.add(ids=ids, embeddings=batch_embeddings, metadatas=metadatas)
        elapsed = time.time() - start_time
        pct = (end / total) * 100
        logger.info(
            f"Batch {batch_num+1}/{batches} — {end:,}/{total:,} ({pct:.1f}%) — {elapsed:.1f}s"
        )

    final_count = coll.count()
    logger.info(
        f"✅ Done! {final_count:,} records indexed in {time.time() - start_time:.1f}s"
    )
    logger.info(f"Vectorstore saved to: {VECTORSTORE_DIR}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    global VECTORSTORE_DIR
    parser = argparse.ArgumentParser(
        description="Ingest watches CSV into ChromaDB"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to watches CSV",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing vectorstore and rebuild",
    )

    parser.add_argument(
        "--vectorstore-dir",
        default=VECTORSTORE_DIR,
        help="Vectorstore output directory",
    )

    args = parser.parse_args()
    VECTORSTORE_DIR = args.vectorstore_dir

    df = load_csv(args.csv)

    build_vectorstore(df, rebuild=args.rebuild)


if __name__ == "__main__":
    main()