import sys
import subprocess

from config import INDEX_FILE, METADATA_FILE

BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
RESET = "\033[0m"


def main():
    print(f"{BOLD}{GREEN}SarvamRAG — Sovereign Document RAG Demo{RESET}\n")

    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        print("No index found. Ingesting sample_docs/sample_printed.txt...")
        result = subprocess.run(
            [sys.executable, "ingest.py", "--text", "sample_docs/sample_printed.txt"],
            cwd=str(INDEX_FILE.parent.parent.parent),
        )
        if result.returncode != 0:
            print("Ingestion failed. Exiting.")
            sys.exit(1)

    from query import load_index, run_query

    index, metadata = load_index()
    num_docs = len(set(c["source_file"] for c in metadata))
    print(f"Index loaded: {len(metadata)} chunks from {num_docs} document(s)\n")

    try:
        while True:
            try:
                user_input = input(f"{CYAN}>{RESET} ").strip()
            except EOFError:
                break
            if not user_input or user_input.lower() in ("quit", "exit"):
                break
            run_query(user_input, index, metadata)
    except KeyboardInterrupt:
        pass

    print("\nBye!")


if __name__ == "__main__":
    main()
