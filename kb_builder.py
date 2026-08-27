from pypdf import PdfReader
import os
import re
import tiktoken
import json
from storage import save_kb_to_s3

from embedder_client import generate_embeddings

# -------------------------------
# PDF READING
# -------------------------------
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    return text

def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# -------------------------------
# TEXT CHUNKING
# -------------------------------
def chunk_text(text, max_tokens=400):

    enc = tiktoken.get_encoding("cl100k_base")

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        if not sentence.strip():
            continue

        sentence_tokens = enc.encode(sentence)

        # If single sentence itself is too large → force split
        if len(sentence_tokens) > max_tokens:
            for i in range(0, len(sentence_tokens), max_tokens):
                chunk_tokens = sentence_tokens[i:i + max_tokens]
                chunks.append(enc.decode(chunk_tokens))
            continue

        # If fits → add to current chunk
        if current_tokens + len(sentence_tokens) <= max_tokens:
            current_chunk += " " + sentence
            current_tokens += len(sentence_tokens)
        else:
            # Save current chunk
            if current_chunk:
                chunks.append(current_chunk.strip())

            current_chunk = sentence
            current_tokens = len(sentence_tokens)

    # Add last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# -------------------------------
# SEMANTIC TEXT CHUNKING
# -------------------------------

def _normalize_text(text):
    """
    Normalize common PDF/text extraction artefacts without
    removing substantive document content.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")

    # Remove literal escaped newlines if present in extracted text
    text = text.replace("\\n", "\n")

    # Normalize horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _looks_like_page_number(line):
    """
    Detect simple standalone page numbers.
    Conservative by design.
    """

    line = line.strip()

    if not line:
        return False

    return bool(
        re.fullmatch(
            r"(?:page\s*)?\d{1,4}",
            line,
            flags=re.IGNORECASE
        )
    )


def _remove_repeated_lines(text, min_occurrences=3):
    """
    Remove repeated document noise such as headers, footers,
    and standalone page numbers.

    Handles both:
        1. standalone repeated lines
        2. repeated header/footer text embedded inside another line

    This remains document-agnostic.
    """

    lines = text.split("\n")

    # --------------------------------
    # STEP 1: NORMALIZE LINES FOR COUNTING
    # --------------------------------

    normalized_lines = []

    for line in lines:

        cleaned = re.sub(
            r"\s+",
            " ",
            line.strip()
        ).lower()

        if cleaned:
            normalized_lines.append(cleaned)

    # --------------------------------
    # STEP 2: FIND FREQUENT LINES
    # --------------------------------

    counts = {}

    for line in normalized_lines:
        counts[line] = counts.get(line, 0) + 1

    repeated = {
        line
        for line, count in counts.items()
        if count >= min_occurrences
        and len(line) >= 20
    }

    # --------------------------------
    # STEP 3: KEEP ONLY PLAUSIBLE
    # DOCUMENT-WIDE NOISE
    # --------------------------------

    # Very short repeated strings are dangerous:
    # words such as "the", "house", "sign", etc.
    #
    # We therefore only treat reasonably long repeated
    # strings as possible headers/footers.

    cleaned_lines = []

    for line in lines:

        original = line
        normalized = re.sub(
            r"\s+",
            " ",
            original.strip()
        ).lower()

        # --------------------------------
        # STANDALONE PAGE NUMBER
        # --------------------------------

        if _looks_like_page_number(original):
            continue

        # --------------------------------
        # STANDALONE REPEATED LINE
        # --------------------------------

        if normalized in repeated:
            continue

        # --------------------------------
        # EMBEDDED REPEATED NOISE
        # --------------------------------

        cleaned = original

        for repeated_line in sorted(
            repeated,
            key=len,
            reverse=True
        ):

            # Convert the normalized repeated line into
            # a whitespace-tolerant regular expression.
            words = repeated_line.split()

            if len(words) < 4:
                continue

            pattern = r"\s+".join(
                re.escape(word)
                for word in words
            )

            cleaned = re.sub(
                pattern,
                " ",
                cleaned,
                flags=re.IGNORECASE
            )

        # --------------------------------
        # FINAL SPACE CLEANUP
        # --------------------------------

        cleaned = re.sub(
            r"[ \t]+",
            " ",
            cleaned
        ).strip()

        if cleaned:
            cleaned_lines.append(cleaned)

    return "\n".join(cleaned_lines)


def _looks_like_heading(line):
    """
    Detect likely structural headings.

    This is intentionally conservative and document-agnostic.
    It uses multiple signals rather than relying on capitalization
    alone.
    """

    line = line.strip()

    if not line:
        return False

    # --------------------------------
    # BASIC GUARDS
    # --------------------------------

    # Very long lines are unlikely to be headings.
    if len(line) > 120:
        return False

    words = line.split()

    if not words:
        return False

    # --------------------------------
    # NUMBERED HEADINGS
    # --------------------------------

    # Examples:
    # 1. Introduction
    # 2. Planetary Aspects
    # 2.1 The Sun
    # 3) Conclusion
    if re.match(
        r"^\d+(?:\.\d+)*[\s.)-]+[A-Za-z]",
        line
    ):
        return True

    # --------------------------------
    # EXPLICIT STRUCTURAL HEADINGS
    # --------------------------------

    if re.match(
        r"^(chapter|section|part|appendix|volume|unit)"
        r"\s+[\w\dIVXivx.-]+",
        line,
        re.IGNORECASE
    ):
        return True

    # --------------------------------
    # KNOWN GENERIC STRUCTURAL LABELS
    # --------------------------------

    if re.match(
        r"^(dec(an)?ate|house|planet|sign|"
        r"introduction|conclusion|summary|"
        r"references|bibliography|contents|"
        r"foreword|preface|index)\b",
        line,
        re.IGNORECASE
    ):
        return True

    # --------------------------------
    # ALL-CAPS HEADINGS
    # --------------------------------

    letters = re.sub(
        r"[^A-Za-z]",
        "",
        line
    )

    if (
        letters
        and len(letters) >= 3
        and line.upper() == line
        and len(words) <= 12
    ):
        return True

    # --------------------------------
    # SHORT TITLE-CASE HEADINGS
    # --------------------------------

    # We only use this as a weak signal.
    # Avoid treating ordinary sentences as headings.
    if 1 <= len(words) <= 6:

        # A sentence ending in punctuation is usually prose.
        if re.search(r"[.!?]$", line):
            return False

        # Count words beginning with uppercase letters.
        title_case_words = sum(
            1
            for word in words
            if word[:1].isupper()
        )

        # Most words start with uppercase → likely title.
        if title_case_words >= max(1, len(words) - 1):

            # Avoid obvious prose-like constructions.
            prose_starters = {
                "These",
                "This",
                "The",
                "They",
                "Their",
                "There",
                "When",
                "Where",
                "Which",
                "Because",
                "Although",
                "However",
                "Once",
                "If",
            }

            if words[0] not in prose_starters:
                return True

    return False


def _reconstruct_line_fragments(text):
    """
    Reconstruct line fragments created by PDF extraction.

    The function is document-agnostic.

    It handles:
    - normal wrapped prose
    - continuation lines
    - numbered/table-like records
    - wrapped table cells
    - strong structural headings

    It does not attempt to reproduce the original visual
    table layout.
    """

    if not text:
        return ""

    lines = text.split("\n")

    reconstructed = []
    current = []

    # --------------------------------
    # HELPERS
    # --------------------------------

    def flush_current():
        nonlocal current

        if not current:
            return

        block = " ".join(
            part.strip()
            for part in current
            if part.strip()
        ).strip()

        if block:
            reconstructed.append(block)

        current = []

    def is_numbered_record(line):
        """
        Detect a new numbered/table record.

        Examples:
            1 Mesha Aries ...
            8 Vrischika Scorpio ...
            12 Meena Pisces ...
        """

        return bool(
            re.match(
                r"^\d{1,3}\s+[A-Za-z]",
                line
            )
        )

    def is_strong_heading(line):
        """
        Strong structural heading detection.

        This intentionally avoids the weaker title-case
        heuristic because short table values can resemble
        headings.
        """

        if not line:
            return False

        # Numbered headings such as:
        # 1. Introduction
        # 2.1 Methods
        if re.match(
            r"^\d+(?:\.\d+)*[\s.)-]+[A-Za-z]",
            line
        ):
            return True

        # Explicit structural headings.
        if re.match(
            r"^(chapter|section|part|appendix|volume|unit)"
            r"\s+[\w\dIVXivx.-]+",
            line,
            re.IGNORECASE
        ):
            return True

        # Generic structural labels.
        if re.match(
            r"^(dec(an)?ate|house|planet|sign|"
            r"introduction|conclusion|summary|"
            r"references|bibliography|contents|"
            r"foreword|preface|index)\b",
            line,
            re.IGNORECASE
        ):
            return True

        # Short all-caps headings.
        letters = re.sub(
            r"[^A-Za-z]",
            "",
            line
        )

        words = line.split()

        if (
            letters
            and len(letters) >= 3
            and line.upper() == line
            and len(words) <= 12
        ):
            return True

        return False

    # --------------------------------
    # STATE
    # --------------------------------

    inside_numbered_record = False

    # --------------------------------
    # MAIN LOOP
    # --------------------------------

    for raw_line in lines:

        line = raw_line.strip()

        # --------------------------------
        # BLANK LINE
        # --------------------------------

        if not line:

            # A numbered/table record may continue across
            # blank lines because PDF extraction often
            # separates cells visually.
            if inside_numbered_record:
                continue

            flush_current()

            if (
                not reconstructed
                or reconstructed[-1] != ""
            ):
                reconstructed.append("")

            continue

        # --------------------------------
        # NEW NUMBERED RECORD
        # --------------------------------

        if is_numbered_record(line):

            # Close previous record.
            flush_current()

            current = [line]

            inside_numbered_record = True

            continue

        # --------------------------------
        # STRONG HEADING
        # --------------------------------

        if is_strong_heading(line):

            # A heading terminates any active record.
            flush_current()

            reconstructed.append(line)

            inside_numbered_record = False

            continue

        # --------------------------------
        # CONTINUATION OF NUMBERED RECORD
        # --------------------------------

        if inside_numbered_record:

            current.append(line)

            continue

        # --------------------------------
        # NORMAL FRAGMENT
        # --------------------------------

        if not current:

            current = [line]

            continue

        previous = current[-1].strip()

        # --------------------------------
        # EXPLICIT CONTINUATION
        # --------------------------------

        # Hyphen/dash at the end means the next
        # line is almost certainly a continuation.
        if previous.endswith(
            ("-", "–", "—")
        ):

            current.append(line)

            continue

        # A line beginning with lowercase text is
        # almost certainly a continuation.
        if re.match(
            r"^[a-z]",
            line
        ):

            current.append(line)

            continue

        # --------------------------------
        # SENTENCE COMPLETION
        # --------------------------------

        if re.search(
            r"[.!?][\"'”’)]?$",
            previous
        ):

            flush_current()

            current = [line]

            continue

        # --------------------------------
        # WRAPPED FRAGMENT
        # --------------------------------

        previous_words = previous.split()
        current_words = line.split()

        if (
            len(previous_words) <= 8
            or len(current_words) <= 8
        ):

            current.append(line)

            continue

        # --------------------------------
        # DEFAULT
        # --------------------------------

        current.append(line)

    # --------------------------------
    # FINAL FLUSH
    # --------------------------------

    flush_current()

    # --------------------------------
    # CLEAN BLANK-LINE DUPLICATES
    # --------------------------------

    cleaned = []

    for item in reconstructed:

        if item == "":

            if (
                cleaned
                and cleaned[-1] != ""
            ):
                cleaned.append("")

            continue

        cleaned.append(item)

    return "\n".join(cleaned)


def _split_into_semantic_blocks(text):
    """
    Split extracted text into structural blocks.

    Major headings create hard boundaries.
    Blank lines create softer paragraph boundaries.
    """

    lines = text.split("\n")

    blocks = []
    current = []

    def flush():

        nonlocal current

        if current:

            block = "\n".join(current).strip()

            if block:
                blocks.append(block)

        current = []

    for line in lines:

        stripped = line.strip()

        if not stripped:

            flush()
            continue

        # A structural heading closes the previous block.
        if _looks_like_heading(stripped):

            flush()

            blocks.append(
                stripped
            )

            continue

        current.append(
            stripped
        )

    flush()

    return blocks


def _clean_block(block):
    """
    Convert PDF line wrapping inside a block into normal prose.
    """

    block = block.strip()

    if not block:
        return ""

    # Join internal line breaks.
    block = re.sub(
        r"\s*\n\s*",
        " ",
        block
    )

    # Normalize spaces.
    block = re.sub(
        r"\s+",
        " ",
        block
    )

    return block.strip()


def _split_oversized_paragraph(
    text,
    max_tokens,
    enc,
    overlap_tokens=50
):
    """
    Split an oversized semantic block.

    Priority:
        paragraph
        -> sentence
        -> token split only as final fallback

    A small overlap is used only for genuine token-level splits.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9\"“'(])",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    pieces = []

    current = []
    current_tokens = 0

    for sentence in sentences:

        sentence_tokens = len(
            enc.encode(sentence)
        )

        # ---------------------------------
        # Single sentence exceeds maximum
        # ---------------------------------
        if sentence_tokens > max_tokens:

            if current:

                pieces.append(
                    " ".join(current).strip()
                )

                current = []
                current_tokens = 0

            tokens = enc.encode(sentence)

            step = max(
                1,
                max_tokens - overlap_tokens
            )

            start = 0

            while start < len(tokens):

                end = min(
                    start + max_tokens,
                    len(tokens)
                )

                piece = enc.decode(
                    tokens[start:end]
                ).strip()

                if piece:
                    pieces.append(piece)

                if end >= len(tokens):
                    break

                start += step

            continue

        # ---------------------------------
        # Sentence fits current piece
        # ---------------------------------
        if (
            current_tokens + sentence_tokens
            <= max_tokens
        ):

            current.append(sentence)
            current_tokens += sentence_tokens

        else:

            if current:

                pieces.append(
                    " ".join(current).strip()
                )

            current = [sentence]
            current_tokens = sentence_tokens

    if current:

        pieces.append(
            " ".join(current).strip()
        )

    return pieces


def semantic_chunk_text(
    text,
    target_tokens=400,
    max_tokens=550,
    min_tokens=80
):
    """
    Create semantically coherent chunks from extracted text.

    Important principles:

    1. Clean obvious extraction noise.
    2. Preserve document structure.
    3. Do not arbitrarily merge major sections.
    4. Prefer paragraph boundaries.
    5. Use sentence boundaries for oversized paragraphs.
    6. Token splitting is the final fallback only.
    """

    if not text:
        return []

    enc = tiktoken.get_encoding(
        "cl100k_base"
    )

    # --------------------------------
    # STEP 1: NORMALIZE
    # --------------------------------

    text = _normalize_text(text)

    if not text:
        return []

    # --------------------------------
    # STEP 2: REMOVE REPEATED NOISE
    # --------------------------------

    text = _remove_repeated_lines(
        text,
        min_occurrences=3
    )

    text = _normalize_text(text)

    # --------------------------------
    # STEP 3: RECONSTRUCT PDF FRAGMENTS
    # --------------------------------

    text = _reconstruct_line_fragments(text)

    text = _normalize_text(text)

    # --------------------------------
    # STEP 4: STRUCTURAL BLOCKS
    # --------------------------------

    raw_blocks = _split_into_semantic_blocks(
        text
    )

    blocks = []

    for block in raw_blocks:

        cleaned = _clean_block(block)

        if cleaned:
            blocks.append(cleaned)

    if not blocks:
        return []

    # --------------------------------
    # STEP 4: BUILD CHUNKS
    # --------------------------------

    chunks = []

    current_parts = []
    current_tokens = 0

    pending_heading = None


    for block in blocks:

        block_tokens = len(
            enc.encode(block)
        )

        # --------------------------------
        # HEADING / STRUCTURAL BLOCK
        # --------------------------------

        if _looks_like_heading(block):

            # Flush any existing content first.
            if current_parts:

                chunks.append(
                    "\n\n".join(
                        current_parts
                    ).strip()
                )

                current_parts = []
                current_tokens = 0

            # Store the heading separately.
            #
            # It will be attached to the FIRST
            # following content block.
            pending_heading = block

            continue


        # --------------------------------
        # OVERSIZED BLOCK
        # --------------------------------

        if block_tokens > max_tokens:

            # Attach pending heading to the beginning
            # of the oversized section when possible.
            if pending_heading:

                heading_tokens = len(
                    enc.encode(pending_heading)
                )

                remaining_tokens = (
                    max_tokens - heading_tokens
                )

                if remaining_tokens > 0:

                    oversized = (
                        _split_oversized_paragraph(
                            block,
                            remaining_tokens,
                            enc
                        )
                    )

                    if oversized:

                        first_chunk = (
                            pending_heading
                            + "\n\n"
                            + oversized[0]
                        )

                        chunks.append(
                            first_chunk.strip()
                        )

                        chunks.extend(
                            oversized[1:]
                        )

                        pending_heading = None

                        continue

                # Heading itself could not be safely
                # attached.
                chunks.append(
                    pending_heading.strip()
                )

                pending_heading = None

            else:

                oversized = (
                    _split_oversized_paragraph(
                        block,
                        max_tokens,
                        enc
                    )
                )

                chunks.extend(
                    oversized
                )

                continue


        # --------------------------------
        # FIRST CONTENT BLOCK
        # --------------------------------

        if pending_heading:

            current_parts = [
                pending_heading,
                block
            ]

            current_tokens = (
                len(enc.encode(pending_heading))
                + block_tokens
            )

            pending_heading = None

            continue


        # --------------------------------
        # FIRST NORMAL BLOCK
        # --------------------------------

        if not current_parts:

            current_parts = [block]
            current_tokens = block_tokens

            continue


        # --------------------------------
        # ADD TO CURRENT CHUNK
        # --------------------------------

        combined_tokens = (
            current_tokens
            + block_tokens
        )

        if combined_tokens <= target_tokens:

            current_parts.append(block)
            current_tokens = combined_tokens

            continue


        # --------------------------------
        # CURRENT CHUNK IS COMPLETE
        # --------------------------------

        if current_tokens >= min_tokens:

            chunks.append(
                "\n\n".join(
                    current_parts
                ).strip()
            )

            current_parts = [block]
            current_tokens = block_tokens

        else:

            # Avoid tiny orphan chunks.
            current_parts.append(block)
            current_tokens = combined_tokens


    # --------------------------------
    # FLUSH PENDING HEADING
    # --------------------------------

    if pending_heading:

        if current_parts:

            combined = (
                pending_heading
                + "\n\n"
                + "\n\n".join(current_parts)
            )

            if len(enc.encode(combined)) <= max_tokens:

                current_parts = [
                    combined
                ]

            else:

                chunks.append(
                    pending_heading.strip()
                )

        else:

            current_parts = [
                pending_heading
            ]


    # --------------------------------
    # STEP 5: FINAL CHUNK
    # --------------------------------

    if current_parts:

        chunks.append(
            "\n\n".join(
                current_parts
            ).strip()
        )

    # --------------------------------
    # STEP 6: MERGE SMALL ORPHAN CHUNKS
    #         WITHOUT CROSSING HEADINGS
    # --------------------------------

    clean_chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk and chunk.strip()
    ]

    small_chunk_tokens = 100

    def starts_with_structural_heading(chunk):
        """
        Determine whether a chunk starts with a structural heading.

        Since headings were deliberately kept as the first block of
        their following content during Step 4, this lets us protect
        section boundaries during orphan merging.
        """

        if not chunk:
            return False

        first_line = chunk.split("\n\n", 1)[0].strip()

        return _looks_like_heading(first_line)


    merged_chunks = []

    for chunk in clean_chunks:

        chunk_tokens = len(
            enc.encode(chunk)
        )

        # --------------------------------
        # NORMAL-SIZED CHUNK
        # --------------------------------

        if chunk_tokens >= small_chunk_tokens:

            merged_chunks.append(chunk)

            continue


        # --------------------------------
        # SMALL CHUNK
        # --------------------------------

        if not merged_chunks:

            # First chunk has nothing before it.
            # Keep it for now.
            merged_chunks.append(chunk)

            continue


        previous = merged_chunks[-1]


        # --------------------------------
        # PROTECT STRUCTURAL BOUNDARIES
        # --------------------------------

        if starts_with_structural_heading(chunk):

            # A small structural heading belongs with the
            # content that follows it, not with the previous
            # section.
            #
            # Example:
            #
            #     CAPRICORN
            #     [Capricorn content...]
            #
            # Keep the heading separate for now so that the
            # following chunk can be attached to it.

            merged_chunks.append(chunk)

            continue


        # --------------------------------
        # SAFE SAME-SECTION MERGE
        # --------------------------------

        combined = (
            previous
            + "\n\n"
            + chunk
        )

        combined_tokens = len(
            enc.encode(combined)
        )


        if combined_tokens <= max_tokens:

            merged_chunks[-1] = combined

        else:

            merged_chunks.append(chunk)


    # --------------------------------
    # STEP 7: SECOND PASS FOR REMAINING
    #         SMALL FIRST / ORPHAN CHUNKS
    # --------------------------------

    final_chunks = []

    for chunk in merged_chunks:

        chunk_tokens = len(
            enc.encode(chunk)
        )


        if (
            final_chunks
            and chunk_tokens < small_chunk_tokens
            and not starts_with_structural_heading(chunk)
        ):

            previous = final_chunks[-1]

            combined = (
                previous
                + "\n\n"
                + chunk
            )

            combined_tokens = len(
                enc.encode(combined)
            )


            if combined_tokens <= max_tokens:

                final_chunks[-1] = combined

            else:

                final_chunks.append(chunk)

        else:

            final_chunks.append(chunk)


    # --------------------------------
    # STEP 8: FINAL SANITY FILTER
    # --------------------------------

    final_chunks = [
        chunk.strip()
        for chunk in final_chunks
        if chunk and chunk.strip()
    ]

    return final_chunks


def chunk_json_text(text, max_tokens=400):

    enc = tiktoken.get_encoding("cl100k_base")

    tokens = enc.encode(text)

    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunks.append(
            enc.decode(tokens[i:i + max_tokens])
        )

    return chunks


# -------------------------------
# OPENAI SETUP
# -------------------------------
# Moved to openai_client.py


# -------------------------------
# EMBEDDINGS
# -------------------------------
def create_embeddings(chunks):
    BATCH_SIZE = 50
    all_embeddings = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        all_embeddings.extend(
            generate_embeddings(batch)
        )

    return all_embeddings


# -------------------------------
# BUILD KB
# -------------------------------
def build_kb(chunks, embeddings):

    kb = []

    for chunk, embedding in zip(chunks, embeddings):

        kb.append({
            "text": chunk,
            "embedding": embedding
        })

    return kb


# -------------------------------
# SAVE KB (UPDATED WITH METADATA)
# -------------------------------
def save_kb(kb, file_id):
    
    # ensure kb folder exists
    os.makedirs("kb", exist_ok=True)
    
    file_path = f"kb/{file_id}.json"

    # 🔥 NEW: metadata calculation
    num_chunks = len(kb)

    # Approximate embedding size (assuming ~4 bytes per float)
    estimated_embedding_size_bytes = sum(len(item["embedding"]) * 4 for item in kb)

    # 🔥 NEW: wrap with metadata
    kb_with_metadata = {
        "metadata": {
            "num_chunks": num_chunks,
            "estimated_embedding_size_bytes": estimated_embedding_size_bytes
        },
        "data": kb
    }

    # Save locally (optional)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(kb_with_metadata, f, indent=2, ensure_ascii=False)

    # 🔥 Upload to S3
    save_kb_to_s3(kb_with_metadata, file_id)