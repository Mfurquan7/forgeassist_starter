
"""Database-only retrieval for ForgeAssist AI Phase 1."""

import re
from dataclasses import dataclass

from core.models import ManualChunk


INSUFFICIENT_EVIDENCE = (
    "I could not find sufficient supporting information in the selected "
    "machine's manuals. Please consult an authorized maintenance engineer "
    "or the manufacturer."
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do",
    "does", "for", "from", "how", "i", "in", "is", "it", "of", "on",
    "or", "should", "the", "this", "to", "what", "when", "where",
    "which", "who", "why", "with", "would", "you", "your", "many",
    "much", "please", "tell", "me",
}


@dataclass
class RetrievedPassage:
    manual_id: int
    manual_title: str
    revision: str
    page_number: int
    text: str
    score: float | None = None


def tokenize(text):
    """Normalize text into meaningful word tokens."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    }


def retrieve_for_machine(machine_id: int, question: str, top_k: int = 4):
    """Search only the selected machine's processed manuals."""
    query_terms = tokenize(question)

    if not query_terms:
        return []

    # Security boundary: search only the selected machine's manuals.
    chunks = (
        ManualChunk.objects
        .filter(
            manual__machine_id=machine_id,
            manual__processing_status__in=["processed", "ready"],
        )
        .select_related("manual")
    )

    ranked = []

    for chunk in chunks:
        chunk_terms = tokenize(chunk.text)
        overlap = query_terms.intersection(chunk_terms)

        if not overlap:
            continue

        score = len(overlap) / len(query_terms)

        ranked.append(
            RetrievedPassage(
                manual_id=chunk.manual_id,
                manual_title=chunk.manual.title,
                revision=chunk.manual.revision,
                page_number=chunk.page_number,
                text=chunk.text,
                score=score,
            )
        )

    ranked.sort(key=lambda passage: passage.score or 0, reverse=True)

    return [
        passage
        for passage in ranked
        if (passage.score or 0) >= 0.20
    ][:max(1, min(top_k, 10))]


def extract_direct_answer(question: str, passage_text: str) -> str:
    """
    Extract a concise answer for common factual questions.
    Falls back to relevant sentences from the retrieved passage.
    This does not use an external AI API.
    """
    question_lower = question.lower()
    text = re.sub(r"\s+", " ", passage_text).strip()

    # Handle document ID questions.
    if "document id" in question_lower:
        match = re.search(
            r"Document ID\s*:\s*([A-Za-z0-9_-]+)",
            text,
            re.IGNORECASE,
        )
        if match:
            return f"The document ID is {match.group(1)}."

    # Handle machine model number questions.
    if "model number" in question_lower:
        match = re.search(
            r"Model number\s*:\s*([A-Za-z0-9_-]+)",
            text,
            re.IGNORECASE,
        )
        if match:
            return f"The model number is {match.group(1)}."

    # Handle manual revision questions.
    if "revision" in question_lower or "revision number" in question_lower:
        match = re.search(
            r"(?:Manual )?revision\s*:\s*([^.;]+)",
            text,
            re.IGNORECASE,
        )
        if match:
            return f"The manual revision is {match.group(1).strip()}."

    # Find relevant sentences for other questions.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    question_terms = tokenize(question)

    scored_sentences = []

    for sentence in sentences:
        sentence_terms = tokenize(sentence)
        overlap = question_terms.intersection(sentence_terms)

        if overlap:
            score = len(overlap) / max(1, len(question_terms))
            scored_sentences.append((score, sentence.strip()))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)

    if scored_sentences:
        # Return up to two relevant sentences, without the whole page.
        selected = [
            sentence for score, sentence in scored_sentences[:2]
            if score > 0
        ]
        answer = " ".join(selected)

        if len(answer) > 500:
            answer = answer[:497].rsplit(" ", 1)[0] + "..."

        return answer

    # Fallback: provide a short excerpt rather than the entire page.
    if len(text) > 350:
        return text[:347].rsplit(" ", 1)[0] + "..."

    return text


def generate_grounded_answer(
    question: str,
    passages: list[RetrievedPassage],
):
    """
    Return a concise answer grounded in retrieved manual text.
    No external AI API is used.
    """
    if not passages:
        return INSUFFICIENT_EVIDENCE, True

    best = passages[0]
    direct_answer = extract_direct_answer(question, best.text)

    answer = (
        f"{direct_answer}\n\n"
        f"Source: {best.manual_title}, revision {best.revision}, "
        f"page {best.page_number}."
    )

    return answer, False
