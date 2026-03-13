"""
WatchScanner - Core RAG Pipeline
Flow:
  1. Resize image  (reduce tokens)
  2. Vision LLM   → extract only visual features (no hallucinated IDs)
  3. Vector search → top-K watch records from 28k CSV-based ChromaDB
  4. Grounded LLM  → identify + price from retrieved context ONLY
"""
import base64
import os
import io
import json
import logging
from typing import Optional

from openai import OpenAI
from PIL import Image
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from app.models import ScanResponse, WatchCandidate

logger = logging.getLogger(__name__)

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./vectorstore")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_IMAGE_PX = int(os.getenv("MAX_IMAGE_PX", "1024"))
COLLECTION_NAME = "watches"


# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

VISION_PROMPT = """
You are a watch image analyst. Your ONLY job is to describe what you SEE in this image.

List exactly these attributes (use "unknown" if not visible):
- Case shape: (round / square / rectangular / tonneau)
- Case color/finish: (silver / gold / rose gold / black PVD / two-tone)
- Dial color: (black / white / blue / green / grey / etc.)
- Dial texture: (plain / sunburst / guilloché / skeleton / etc.)
- Bezel type: (smooth / fluted / tachymeter / diver / ceramic / etc.)
- Bezel color:
- Hour markers: (indices / roman / arabic numerals / mixed)
- Hands style: (dauphine / baton / sword / skeleton / mercedes)
- Crown position: (3 o'clock / 4 o'clock / 12 o'clock / left side)
- Complications visible: (date / chronograph / moon phase / GMT / none)
- Date window: (yes at X / no)
- Bracelet/strap type: (oyster / jubilee / leather / rubber / NATO / mesh)
- Bracelet/strap color:
- Any visible text or logo on dial:
- Case size estimate: (small <36mm / medium 36-40mm / large 40-44mm / XL >44mm)

DO NOT guess the brand or model. Only describe visual facts.
"""

GROUNDING_PROMPT = """
You are an expert watch identification system. You must be strictly factual.

## VISUAL OBSERVATIONS FROM IMAGE
{visual_features}

## TOP MATCHING WATCH RECORDS FROM DATABASE
{context}

## TASK
Using ONLY the visual observations and the database records above:

1. **Identify** the most likely watch (brand, model, reference number)
2. **Justify** each identification point by matching a visual feature to a database field
3. **Project price** based on the price field in the database records
4. **Estimate price range** (min - max across matching records)
5. **Rate confidence**: High (strong visual+DB match), Medium (partial match), Low (weak match)
6. **Explain confidence** in one sentence

Respond in this EXACT JSON format (no markdown, no backticks):
{{
  "identified_brand": "...",
  "identified_model": "...",
  "identified_ref": "...",
  "projected_price": "...",
  "price_range": "... - ...",
  "price_basis": "Based on N matching records",
  "confidence": "High|Medium|Low",
  "confidence_reason": "...",
  "movement": "...",
  "bracelet": "...",
  "year_of_production": "...",
  "analysis": "..."
}}

RULES:
- projected_price and price_range must come from the database records only
- If you cannot determine a field, use "Unknown"  
- Never invent specifications not present in the retrieved records
- If no good match is found, set confidence to "Low" and explain why
"""


# ──────────────────────────────────────────────
# Scanner class
# ──────────────────────────────────────────────

class _Doc:
    """Minimal doc-like object with metadata for retrieval results."""
    __slots__ = ("metadata",)
    def __init__(self, metadata: dict):
        self.metadata = metadata


class WatchScanner:
    def __init__(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        self._client = OpenAI()
        logger.info(f"Vision/text models: {OPENAI_VISION_MODEL}, {OPENAI_TEXT_MODEL}")
        logger.info(f"Loading Hugging Face embeddings: {EMBED_MODEL}")
        self.embeddings = SentenceTransformer(EMBED_MODEL)
        self._chroma = PersistentClient(path=VECTORSTORE_DIR)
        self._coll = self._chroma.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        count = self._coll.count()
        logger.info(f"Vectorstore loaded — {count:,} watch records indexed")

    # ── Step 1: Resize image ──────────────────
    def _resize_image(self, image_bytes: bytes) -> Image.Image:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_IMAGE_PX:
            ratio = MAX_IMAGE_PX / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            logger.info(f"Resized image from {w}x{h} → {img.size}")
        return img

    # ── Step 2: Extract visual features ──────
    def _extract_visual_features(self, image_bytes: bytes) -> str:
        img = self._resize_image(image_bytes)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        response = self._client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            max_tokens=400,
            temperature=0.1,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
        )
        features = (response.choices[0].message.content or "").strip()
        logger.info(f"Visual features extracted ({len(features)} chars)")
        return features

    # ── Step 3: Vector retrieval ─────────────
    def _retrieve_candidates(self, visual_description: str) -> list:
        emb = self.embeddings.encode(
            [visual_description], normalize_embeddings=True
        )[0].tolist()
        results = self._coll.query(
            query_embeddings=[emb], n_results=TOP_K, include=["metadatas", "distances"]
        )
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        if not ids:
            logger.info("No candidates from vectorstore (empty or no match)")
            return []
        # Cosine distance → similarity (1 - distance)
        out = [(_Doc(m), max(0.0, 1.0 - d)) for m, d in zip(metadatas, distances)]
        logger.info(f"Retrieved {len(out)} candidates from vectorstore")
        return out

    # ── Step 4: Grounded generation ──────────
    def _generate_identification(self, visual_features: str, candidates: list) -> dict:
        context_parts = []
        for i, (doc, score) in enumerate(candidates):
            m = doc.metadata
            context_parts.append(
                f"RECORD {i+1} (relevance: {score:.2f}):\n"
                f"  Name: {m.get('name','?')} | Brand: {m.get('brand','?')} | "
                f"Model: {m.get('model','?')} | Ref: {m.get('ref','?')}\n"
                f"  Movement: {m.get('mvmt','?')} | Bracelet: {m.get('bracem','?')} | "
                f"Year: {m.get('yop','?')} | Price: {m.get('price','?')}"
            )
        context = "\n\n".join(context_parts)
        prompt = GROUNDING_PROMPT.format(
            visual_features=visual_features,
            context=context,
        )
        response = self._client.chat.completions.create(
            model=OPENAI_TEXT_MODEL,
            max_tokens=600,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (response.choices[0].message.content or "").strip()
        # Strip markdown fences if model adds them
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"JSON parse failed, raw response:\n{raw}")
            result = {
                "identified_brand": "Parse Error",
                "identified_model": "Parse Error",
                "identified_ref": "Unknown",
                "projected_price": "Unknown",
                "price_range": "Unknown",
                "price_basis": "Unknown",
                "confidence": "Low",
                "confidence_reason": "Model returned malformed JSON",
                "movement": "Unknown",
                "bracelet": "Unknown",
                "year_of_production": "Unknown",
                "analysis": raw[:500],
            }
        return result

    # ── Public entry point ────────────────────
    def scan(self, image_bytes: bytes) -> ScanResponse:
        # Step 1+2: Vision → features
        visual_features = self._extract_visual_features(image_bytes)

        # Step 3: RAG retrieval
        candidates = self._retrieve_candidates(visual_features)

        # Step 4: Grounded LLM answer
        result = self._generate_identification(visual_features, candidates)

        top_candidates = []
        for doc, score in candidates:
            m = doc.metadata
            top_candidates.append(WatchCandidate(
                name=m.get("name"),
                brand=m.get("brand"),
                model=m.get("model"),
                ref=m.get("ref"),
                mvmt=m.get("mvmt"),
                bracem=m.get("bracem"),
                yop=m.get("yop"),
                price=m.get("price"),
                similarity_score=round(score, 3),
            ))

        return ScanResponse(
            identified_brand=result.get("identified_brand", "Unknown"),
            identified_model=result.get("identified_model", "Unknown"),
            identified_ref=result.get("identified_ref"),
            visual_features=visual_features,
            projected_price=result.get("projected_price", "Unknown"),
            price_range=result.get("price_range"),
            price_basis=result.get("price_basis"),
            confidence=result.get("confidence", "Low"),
            confidence_reason=result.get("confidence_reason", ""),
            movement=result.get("movement"),
            bracelet=result.get("bracelet"),
            year_of_production=result.get("year_of_production"),
            top_candidates=top_candidates,
            analysis=result.get("analysis", ""),
        )