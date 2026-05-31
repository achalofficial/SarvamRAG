# SarvamRAG — Sovereign Document RAG with Sarvam Vision + TurboVec

*End-to-end document RAG pipeline: Sarvam Vision OCR → TurboVec compressed vector search → local LLM generation. Runs entirely on Apple Silicon.*

---

## What It Does

SarvamRAG lets you chat with scanned documents — handwritten notes, printed PDFs, Indic-script files — using a fully local retrieval and generation stack. Only the OCR step touches the network (Sarvam Vision API). Everything else runs on-device.

1. **Ingest** — Digitize a document via Sarvam Vision, chunk the text, embed with `nomic-embed-text`, store in a 4-bit TurboVec index
2. **Query** — Embed your question, find the top-k chunks via TurboVec, generate an answer with a local LLM

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                        INGEST PIPELINE                          ║
╚══════════════════════════════════════════════════════════════════╝

  Document (PDF / JPG / PNG / TXT)
            │
            ▼
  ┌─────────────────────┐
  │  Sarvam Vision API  │  ← OCR: 23 Indian languages, ₹0.5/page
  │  (remote, optional) │    Use --text to skip for plain text
  └─────────────────────┘
            │  markdown text
            ▼
  ┌─────────────────────┐
  │  Recursive Chunker  │  ← ~500 char chunks, 50 char overlap
  └─────────────────────┘
            │  list of chunks
            ▼
  ┌─────────────────────┐
  │  Ollama Embeddings  │  ← nomic-embed-text, 768-dim, local
  │  (POST /api/embed)  │
  └─────────────────────┘
            │  float32 vectors
            ▼
  ┌─────────────────────┐
  │  TurboVec Index     │  ← 4-bit quantized (TurboQuant)
  │  sarvam_rag.tvec    │    6× compression vs float32
  │  chunks_metadata.json│   zero training, just index.add()
  └─────────────────────┘
            │
         [disk]


╔══════════════════════════════════════════════════════════════════╗
║                        QUERY PIPELINE                           ║
╚══════════════════════════════════════════════════════════════════╝

  User Query (natural language)
            │
            ▼
  ┌─────────────────────┐
  │  Ollama Embeddings  │  ← same nomic-embed-text model
  └─────────────────────┘
            │  query vector (1 × 768)
            ▼
  ┌─────────────────────┐
  │  TurboVec Search    │  ← top-k ANN on 4-bit index
  │  (ARM NEON SIMD)    │    12–20% faster than FAISS on M4
  └─────────────────────┘
            │  chunk indices + scores
            ▼
  ┌─────────────────────┐
  │  Metadata Lookup    │  ← retrieve chunk text + source file
  └─────────────────────┘
            │  RAG prompt
            ▼
  ┌─────────────────────┐
  │  Ollama Generation  │  ← phi3 / gemma3 / any local model
  │  (POST /api/generate│    100% on-device
  └─────────────────────┘
            │
            ▼
  Answer  +  Source Citations
```

---

## Why This Stack

| | |
|--|--|
| **Sarvam Vision** | Handles 23 Indian languages (Devanagari, Tamil, Telugu, Bengali…) at ₹0.5/page — the only OCR engine built specifically for Indic-script documents. 35M+ pages digitized. Standard Tesseract/Google Vision fail on mixed-script and handwritten Indic content. |
| **TurboVec (TurboQuant)** | Google Research, ICLR 2026. Compresses 31 GB of float32 vectors to ~4 GB with **zero training, no codebook, no offline step** — you just call `index.add(vectors)`. Quantizes to 4-bit with Lloyd-Max centroids computed on-the-fly. |
| **ARM SIMD advantage** | TurboVec's blocked memory layout maps directly to Apple Silicon NEON lanes. On M4, it retrieves 12–20% faster than a FAISS flat index at equal recall. The whole index fits in unified memory. |
| **No framework tax** | Pure Python + direct REST calls to Ollama. No LangChain, no LlamaIndex. Every line of the stack is readable; nothing is hidden behind an abstraction layer. |

---

## Project Structure

```
SarvamRAG/
├── config.py              # All constants, loaded from .env
├── chunker.py             # Recursive character text splitter
├── ingest.py              # Ingest pipeline (Sarvam → chunk → embed → TurboVec)
├── query.py               # Query pipeline (embed → search → generate)
├── demo.py                # Interactive colored terminal demo
├── requirements.txt       # sarvamai, turbovec, requests, python-dotenv, numpy
├── .env.example           # Environment variable template
├── sample_docs/
│   ├── sample_printed.txt # Sample text about India's digital infrastructure
│   └── README_SAMPLES.md  # Instructions for adding your own documents
└── data/                  # Runtime (gitignored)
    ├── extracted/         # Sarvam Vision markdown output
    └── index/
        ├── sarvam_rag.tvec        # TurboVec index (4-bit quantized)
        └── chunks_metadata.json   # Parallel chunk text + source metadata
```

---

## Setup

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) installed and running.

```bash
# 1. Clone and install
git clone <repo-url>
cd SarvamRAG
pip install -r requirements.txt

# 2. Pull the embedding model
ollama pull nomic-embed-text

# 3. Pull a generation model (use whatever you have; phi3 and gemma3:4b both work)
ollama pull phi3

# 4. Configure environment
cp .env.example .env
# Open .env and set SARVAM_API_KEY (only needed for PDF/image ingestion)
# Set GEN_MODEL to match your pulled model, e.g. phi3:latest
```

---

## Usage

### Ingest — plain text (free, no API key)

```bash
python ingest.py --text sample_docs/sample_printed.txt
```

```
Chunked sample_printed.txt into 13 chunks
Embedding chunk 1/13...
Embedding chunk 2/13...
...
Embedding chunk 13/13...
Saved index with 13 total vectors
```

### Ingest — scanned document (Sarvam Vision API)

```bash
# Single file
python ingest.py --file invoice.pdf --language en-IN

# Batch — all PDFs/images in a folder
python ingest.py --dir ./my_documents/

# Supported languages: en-IN, hi-IN, ta-IN, te-IN, bn-IN, gu-IN, kn-IN, ml-IN, mr-IN…
```

### Query — single question

```bash
python query.py "What is UPI's role in financial inclusion?"
```

```
[dim] [1] score=0.6152 | sample_docs/sample_printed.txt chunk#0
[dim] [2] score=0.6062 | sample_docs/sample_printed.txt chunk#2
[dim] [3] score=0.6026 | sample_docs/sample_printed.txt chunk#8
[dim] [4] score=0.5887 | sample_docs/sample_printed.txt chunk#1
[dim] [5] score=0.5762 | sample_docs/sample_printed.txt chunk#3

Sources:
  sample_docs/sample_printed.txt

UPI (Unified Payments Interface) has fundamentally transformed financial
inclusion in India by enabling instant bank-to-bank transfers via mobile phone
without requiring a card or internet banking setup...
```

### Query — interactive mode

```bash
python query.py --interactive
```

```
> What is Aadhaar used for?
> How does BharatNet work?
> exit
```

### Demo — all-in-one

Automatically ingests the sample doc if no index exists, then drops into an interactive loop with colored output.

```bash
python demo.py
```

---

## Performance (Mac Mini M4, 16 GB)

| Operation | Latency |
|-----------|---------|
| Embed one chunk (nomic-embed-text) | ~50 ms |
| TurboVec search (top-5, 13 chunks) | < 1 ms |
| Generation (phi3:latest, ~200 tokens) | ~2–4 s |
| Full query round-trip | ~2–5 s |

**Memory footprint:**
- 13 chunks → 11 KB on disk (4-bit TurboVec index)
- 1M chunks (full-scale) → ~3–4 GB vs 3 GB float32 baseline — fits in M4's 16 GB unified memory alongside the LLM

---

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| OCR / Digitization | Sarvam Vision API (`sarvamai`) | SOTA for Indic + English docs |
| Chunking | Custom recursive splitter | Simple, no deps, 500-char target |
| Embedding | Ollama `nomic-embed-text` | 768-dim, local, free |
| Vector Index | `turbovec` (TurboQuant 4-bit) | 6× compression, ARM NEON-optimized |
| Generation | Ollama (any local model) | 100% on-device |
| Orchestration | Pure Python CLI | No LangChain, no LlamaIndex |

---

## License

MIT
