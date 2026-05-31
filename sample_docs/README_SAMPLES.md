# Sample Documents for SarvamRAG Demo

This folder contains sample documents used to demonstrate the RAG (Retrieval-Augmented Generation) pipeline. You can add your own documents here and ingest them into the vector store.

## Supported Formats

| Format | Notes |
|--------|-------|
| JPG, PNG | Scanned images of printed or handwritten text |
| PDF | Recommended max 10 pages per file for reasonable processing time |
| TXT | Plain text files, ingested directly without OCR |

## How to Ingest Documents

### Free Mode — Plain Text (no API cost)

Use this for `.txt` files or pre-extracted text content:

```bash
python ingest.py --text sample_docs/sample_printed.txt
```

This mode skips the Sarvam Vision API and reads the text directly, making it ideal for testing and development.

### Document Scan Mode — Sarvam Vision API

Use this for scanned PDFs, images, or any file requiring OCR:

```bash
python ingest.py --file your_document.pdf
```

> **Note:** Sarvam Vision API is billed at **₹0.5 per page**. A 10-page PDF will cost ₹5. Use `--text` mode for free testing.

## Included Sample Files

- **sample_printed.txt** — Three paragraphs on India's digital infrastructure (UPI, Aadhaar, BharatNet). Use this to verify the full RAG pipeline works end-to-end without any API costs.

## Tips

- Keep PDFs under 10 pages for best results and manageable costs.
- For multi-document ingestion, run `ingest.py` once per file; each run appends to the vector store.
- Delete `chroma_db/` and re-run ingest if you want to start fresh.
