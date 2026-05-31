import argparse
import json
import sys
import numpy as np
import requests
import turbovec

from config import (
    OLLAMA_BASE_URL, EMBED_MODEL, GEN_MODEL, TOP_K,
    INDEX_FILE, METADATA_FILE,
)

DIM = "\033[2m"
RESET = "\033[0m"
YELLOW = "\033[33m"
GREEN = "\033[32m"


def get_embedding(text: str) -> list[float]:
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
    except requests.exceptions.ConnectionError:
        print(f"Error: Ollama not running at {OLLAMA_BASE_URL}")
        sys.exit(1)


def generate(prompt: str) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": GEN_MODEL, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["response"]
    except requests.exceptions.ConnectionError:
        print(f"Error: Ollama not running at {OLLAMA_BASE_URL}")
        sys.exit(1)


def load_index():
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        print("Error: No index found. Run python ingest.py --text <file> first")
        sys.exit(1)
    index = turbovec.TurboQuantIndex.load(str(INDEX_FILE))
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
    return index, metadata


def run_query(query: str, index, metadata: list) -> None:
    embedding = get_embedding(query)
    query_vec = np.array([embedding], dtype=np.float32)
    scores, positions = index.search(query_vec, k=TOP_K)
    retrieved_positions = positions[0].tolist()
    retrieved_scores = scores[0].tolist()

    chunks = [metadata[p] for p in retrieved_positions]

    context_parts = []
    for i, (chunk, score) in enumerate(zip(chunks, retrieved_scores)):
        print(f"{DIM}[{i+1}] score={score:.4f} | {chunk['source_file']} chunk#{chunk['chunk_id']}{RESET}")
        context_parts.append(chunk["text"])

    context_str = "\n\n---\n".join(context_parts)
    prompt = (
        "You are a helpful document assistant. Answer the question using ONLY "
        "the provided context. If the context doesn't contain enough information, say so.\n\n"
        f"Context:\n---\n{context_str}\n\n---\n\nQuestion: {query}\n\nAnswer:"
    )

    print(f"\n{YELLOW}Sources:{RESET}")
    for chunk in chunks:
        print(f"  {chunk['source_file']}")

    answer = generate(prompt)
    print(f"\n{GREEN}{answer}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="Query the SarvamRAG index.")
    parser.add_argument("query", nargs="?", help="Query text")
    parser.add_argument("--interactive", action="store_true", help="Interactive query loop")
    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.print_usage()
        print("Hint: python query.py 'your question' or python query.py --interactive")
        sys.exit(0)

    index, metadata = load_index()

    if args.interactive:
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in ("quit", "exit", ""):
                break
            run_query(user_input, index, metadata)
    else:
        run_query(args.query, index, metadata)


if __name__ == "__main__":
    main()
