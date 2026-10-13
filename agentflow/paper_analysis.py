"""Deterministic, page-linked paper comparison for the no-AI execution mode."""

import re
from collections import Counter


STOP_WORDS = {
    "about", "after", "also", "among", "because", "before", "between", "could",
    "data", "from", "have", "into", "more", "other", "paper", "research",
    "results", "study", "their", "there", "these", "this", "those", "through",
    "using", "were", "which", "while", "with", "would", "your", "that", "than",
    "they", "them", "such", "only", "each", "both", "does", "will", "been",
}


def terms(text):
    return [word for word in re.findall(r"[a-z][a-z0-9-]{3,}", text.lower())
            if word not in STOP_WORDS]


def page_sentences(document):
    for page in document["pages"]:
        normalized = re.sub(r"\s+", " ", page["text"]).strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized):
            sentence = sentence.strip()
            if 35 <= len(sentence) <= 650 and len(terms(sentence)) >= 5:
                yield {"page": page["page"], "text": sentence}


def extractive_summary(document, objective):
    """Return a short selection of original paper sentences with page references."""
    selected = []
    for page in document["pages"][:3]:
        match = re.search(r"\bAbstract\b\s*(.*?)\s*\bKeywords?\s*:",
                          page["text"], flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        abstract = re.sub(r"\s+", " ", match.group(1)).strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", abstract):
            sentence = sentence.strip()
            if 25 <= len(sentence) <= 650:
                selected.append({"page": page["page"], "text": sentence})
        if selected:
            break
    if not selected:
        candidates = list(page_sentences(document))
        objective_terms = set(terms(objective))
        selected = sorted(candidates, key=lambda item: (
            -len(set(terms(item["text"])) & objective_terms), item["page"]
        ))[:4]
    if not selected:
        for page in document["pages"]:
            text = re.sub(r"\s+", " ", page["text"]).strip()
            if text:
                selected = [{"page": page["page"], "text": text[:650]}]
                break
    return selected[:5]


def compare_documents(documents, objective):
    """Select supporting excerpts; never generate claims absent from the files."""
    if len(documents) < 2:
        raise ValueError("At least two documents are required")
    objective_terms = set(terms(objective))
    counts = [Counter(terms(doc["text"])) for doc in documents]
    shared = set(counts[0])
    for count in counts[1:]:
        shared &= set(count)
    shared_terms = sorted(shared, key=lambda word: (-sum(c[word] for c in counts), word))[:12]
    paper_cards = []
    for doc, count in zip(documents, counts):
        candidates = list(page_sentences(doc))
        ranked = sorted(candidates, key=lambda item: (
            -len(set(terms(item["text"])) & objective_terms),
            -len(set(terms(item["text"])) & set(shared_terms)),
            item["page"],
        ))
        selected = ranked[:4]
        paper_cards.append({
            "name": doc["name"],
            "pages": len(doc["pages"]),
            "top_terms": [word for word, _ in count.most_common(8)],
            "evidence": selected,
        })
    return {
        "kind": "comparison",
        "documents": [doc["name"] for doc in documents],
        "shared_terms": shared_terms,
        "papers": paper_cards,
        "method": "Deterministic keyword ranking of extracted sentences; not semantic or factual verification",
    }
