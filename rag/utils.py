# ==========================================
# utils.py
# ==========================================
import re
import json
from pathlib import Path

def normalize_text(text: str) -> str:
    """
    Cleans and normalizes text for consistent embeddings.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,-]', '', text)
    return text

def serialize_metadata(row: dict) -> dict:
    """
    Converts a CSV row or record into clean metadata for ChromaDB.
    """
    return {k: str(v) for k, v in row.items() if v is not None}

def ensure_dir(path: str | Path):
    """
    Ensures a directory exists.
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data, path: str | Path):
    """
    Saves Python objects as JSON files.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path: str | Path):
    """
    Loads JSON files safely.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
