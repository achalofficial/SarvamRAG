import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests
import turbovec

from chunker import chunk_text
from config import (
    SARVAM_API_KEY, OLLAMA_BASE_URL, EMBED_MODEL,
    EMBED_DIM, TURBOVEC_BITS, CHUNK_SIZE, CHUNK_OVERLAP,
    INDEX_FILE, METADATA_FILE, EXTRACTED_DIR,
)


def get_embedding(text: str) -> list[float]:
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
    except requests.exceptions.ConnectionError:
        print(f"Error: Ollama not running at {OLLAMA_BASE_URL}")
        sys.exit(1)


def digitize_with_sarvam(file_path: Path, language: str) -> str:
    from sarvamai import SarvamAI
    client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    print(f"Digitizing {file_path.name}...")
    job = client.document_intelligence.create_job(language=language, output_format="md")
    job.upload_file(str(file_path))
    job.start()
    job.wait_until_complete()
    out_path = EXTRACTED_DIR / (file_path.stem + ".md")
    job.download_output(str(out_path))
    return out_path.read_text(encoding="utf-8")


def collect_files(args) -> list[tuple[Path, str]]:
    files = []
    if args.text:
        p = Path(args.text)
        if not p.exists():
            print(f"Error: File not found: {p}")
            sys.exit(1)
        files.append((p, "text"))
    if args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"Error: File not found: {p}")
            sys.exit(1)
        files.append((p, "vision"))
    if args.dir:
        d = Path(args.dir)
        if not d.exists():
            print(f"Error: Directory not found: {d}")
            sys.exit(1)
        for ext in ("*.pdf", "*.jpg", "*.jpeg", "*.png"):
            for f in sorted(d.glob(ext)):
                files.append((f, "vision"))
    return files


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into SarvamRAG index")
    parser.add_argument("--file", metavar="PATH", help="Single PDF/image via Sarvam Vision")
    parser.add_argument("--dir", metavar="DIR", help="Batch ingest PDF/images in folder")
    parser.add_argument("--text", metavar="PATH", help="Plain text file (skips Sarvam Vision)")
    parser.add_argument("--language", default="en-IN", help="Language code (default: en-IN)")
    args = parser.parse_args()

    if not args.file and not args.dir and not args.text:
        parser.print_help()
        sys.exit(1)

    if (args.file or args.dir) and not SARVAM_API_KEY:
        print("Error: SARVAM_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    files = collect_files(args)
    new_chunks = []

    for file_path, mode in files:
        if mode == "text":
            text = file_path.read_text(encoding="utf-8")
        else:
            text = digitize_with_sarvam(file_path, args.language)

        chunks = chunk_text(text, str(file_path), chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        print(f"Chunked {file_path.name} into {len(chunks)} chunks")
        new_chunks.extend(chunks)

    if not new_chunks:
        print("No chunks produced. Nothing to index.")
        sys.exit(0)

    # Load existing metadata; re-embed stored texts to rebuild combined index
    existing_metadata: list[dict] = []
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            existing_metadata = json.load(f)

    all_chunks = existing_metadata + new_chunks
    total = len(all_chunks)

    all_embeddings = []
    for i, chunk in enumerate(all_chunks, 1):
        print(f"Embedding chunk {i}/{total}...")
        all_embeddings.append(get_embedding(chunk["text"]))

    index = turbovec.TurboQuantIndex(dim=EMBED_DIM, bit_width=TURBOVEC_BITS)
    index.add(np.array(all_embeddings, dtype=np.float32))
    index.prepare()
    index.write(str(INDEX_FILE))

    with open(METADATA_FILE, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Saved index with {total} total vectors")


if __name__ == "__main__":
    main()
