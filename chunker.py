def _split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        # Hard limit: force-split by character
        result = []
        i = 0
        while i < len(text):
            result.append(text[i:i + chunk_size])
            i += chunk_size
        return result

    sep = separators[0]
    rest = separators[1:]

    if sep not in text:
        return _split(text, rest, chunk_size)

    parts = text.split(sep)
    chunks = []
    current = ""

    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(_split(part, rest, chunk_size))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    return chunks if chunks else [text]


def chunk_text(text: str, source_file: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    separators = ["\n\n", "\n", ". ", " "]
    raw_chunks = _split(text, separators, chunk_size)
    raw_chunks = [c for c in raw_chunks if c.strip()]

    if not raw_chunks:
        return []

    # Reconstruct positions by scanning forward through original text
    # Match each chunk against remaining text to find its true offset
    positions = []
    pos = 0
    remaining = text
    for chunk in raw_chunks:
        idx = remaining.find(chunk)
        if idx == -1:
            idx = 0
        abs_start = pos + idx
        abs_end = abs_start + len(chunk)
        positions.append((abs_start, abs_end))
        # Advance past the chunk in remaining text
        advance = idx + len(chunk)
        pos += advance
        remaining = text[pos:]

    result = []
    for i, (start, end) in enumerate(positions):
        if i == 0:
            result.append(text[start:end])
        else:
            # Each chunk (except the first) starts overlap chars before the previous chunk ended
            overlap_start = max(0, positions[i - 1][1] - overlap)
            result.append(text[overlap_start:end])

    result = [c for c in result if c.strip()]
    return [{"source_file": source_file, "chunk_id": i, "text": c} for i, c in enumerate(result)]
