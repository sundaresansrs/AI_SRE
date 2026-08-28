"""
Phase 3, Step 1: Chunk incident postmortem runbooks into 200-400 word passages.

Reads every .txt file in data/runbooks/void/, splits on sentence boundaries
targeting ~300 words per chunk, and writes data/processed/runbook_chunks.parquet.

Files under 200 words total are kept as a single chunk.
The final chunk of any file may be shorter than 200 (remainder).
No chunk may exceed 400 words.
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNBOOK_DIR = REPO_ROOT / "data" / "runbooks" / "void"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "runbook_chunks.parquet"
TARGET_WORDS = 300
MIN_WORDS = 200
MAX_WORDS = 400


# ---------------------------------------------------------------------------
# Sentence splitter (handles common abbreviations, decimals, ellipses)
# ---------------------------------------------------------------------------
_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\d\"\'(])"   # split after .!? followed by space + uppercase/digit/quote
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving whitespace within sentences."""
    sentences = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------
def chunk_text(text: str, target: int = TARGET_WORDS,
               min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> list[str]:
    """
    Split *text* into chunks of target ~300 words on sentence boundaries.

    Rules
    -----
    - If total word count < min_words → return whole text as one chunk.
    - Build chunks by accumulating sentences until adding the next sentence
      would exceed max_words.  If the current chunk has >= min_words, flush it
      and start a new one.  Otherwise keep accumulating (favour oversized
      chunks over mid-sentence splits — but sentences are short so this is
      unlikely to breach max_words significantly).
    - The final chunk may be shorter than min_words (remainder).
    """
    total_wc = len(text.split())
    if total_wc <= min_words:
        return [text.strip()]

    sentences = split_sentences(text)
    if not sentences:
        return [text.strip()]

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_wc = 0

    for sentence in sentences:
        s_wc = len(sentence.split())

        # Would adding this sentence exceed max_words?
        if current_wc + s_wc > max_words and current_wc >= min_words:
            # Flush current chunk
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_wc = s_wc
        else:
            current_sentences.append(sentence)
            current_wc += s_wc

    # Flush remainder
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print("PHASE 3, STEP 1: CHUNK INCIDENT RUNBOOKS")
    print("=" * 80)

    txt_files = sorted(RUNBOOK_DIR.glob("*.txt"))
    print(f"\n1. Found {len(txt_files)} runbook files in {RUNBOOK_DIR}")
    if len(txt_files) != 34:
        raise RuntimeError(f"Expected 34 runbooks, found {len(txt_files)}")

    rows: list[dict] = []
    file_chunk_counts: dict[str, int] = {}

    for fpath in txt_files:
        text = fpath.read_text(encoding="utf-8")
        fname = fpath.name
        stem = fpath.stem

        chunks = chunk_text(text)
        file_chunk_counts[fname] = len(chunks)

        for idx, chunk in enumerate(chunks):
            rows.append({
                "chunk_id": f"{stem}_chunk{idx}",
                "source_file": fname,
                "chunk_text": chunk,
                "word_count": len(chunk.split()),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No chunks were created")

    invalid = df[(df["word_count"] > MAX_WORDS) | (df["word_count"] < 1)]
    if not invalid.empty:
        raise ValueError(
            "Chunk word counts are invalid:\n"
            f"{invalid[['chunk_id', 'word_count']].to_string(index=False)}"
        )

    # Write parquet
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n2. Wrote {len(df)} chunks to {OUTPUT_PATH}")

    # ---- Console report ----
    print(f"\n3. Total chunks created: {len(df)}")

    print("\n4. Per-file breakdown:")
    for fname, count in file_chunk_counts.items():
        print(f"      {fname:60s} -> {count} chunk(s)")

    print("\n5. df.head(10):")
    preview = df.head(10).copy()
    preview["chunk_text_preview"] = preview["chunk_text"].apply(
        lambda t: " ".join(t.split()[:15]) + "..."
    )
    print(preview[["chunk_id", "source_file", "word_count", "chunk_text_preview"]].to_string(index=False))

    # ---- 200-400 rule verification (excluding final-chunk remainders) ----
    # For each file identify whether a chunk is the last one
    is_last = []
    for _, group in df.groupby("source_file"):
        n = len(group)
        is_last.extend([False] * (n - 1) + [True])
    df["_is_last"] = is_last

    non_remainder = df[~df["_is_last"]]
    if len(non_remainder) > 0:
        print(f"\n6. Non-remainder chunk word-count verification:")
        print(f"      min(word_count) = {non_remainder['word_count'].min()}")
        print(f"      max(word_count) = {non_remainder['word_count'].max()}")
    else:
        print("\n6. All chunks are single-file chunks (no non-remainder chunks to verify).")

    # Also show overall stats
    print(f"\n7. Overall word-count stats (all chunks):")
    print(f"      min = {df['word_count'].min()}")
    print(f"      max = {df['word_count'].max()}")
    print(f"      mean = {df['word_count'].mean():.1f}")
    print(f"      median = {df['word_count'].median():.1f}")

    print("\n" + "=" * 80)
    print("CHUNKING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
