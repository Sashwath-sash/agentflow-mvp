from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ResearchSource:
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    url: str | None


class ResearchSearchError(RuntimeError):
    """Raised when the research provider cannot return sources."""


class CrossrefSearchProvider:
    """Small provider adapter for Crossref's public scholarly-metadata API."""

    endpoint = "https://api.crossref.org/works"

    def search(self, query: str, limit: int = 5) -> list[ResearchSource]:
        query = query.strip()
        if not query:
            raise ResearchSearchError("query must not be empty")
        params = urlencode({"query": query, "rows": max(1, min(limit, 20)), "select": "title,author,published,DOI,URL"})
        request = Request(f"{self.endpoint}?{params}", headers={"User-Agent": "AgentFlow/0.1 (academic research prototype)"})
        try:
            with urlopen(request, timeout=15) as response:
                payload = json.load(response)
        except Exception as exc:
            raise ResearchSearchError("research provider request failed") from exc
        sources: list[ResearchSource] = []
        for item in payload.get("message", {}).get("items", []):
            titles = item.get("title") or []
            if not titles:
                continue
            authors = [" ".join(filter(None, [author.get("given"), author.get("family")])) for author in item.get("author", [])]
            date_parts = (item.get("published", {}).get("date-parts") or [[]])[0]
            sources.append(ResearchSource(title=titles[0], authors=authors, year=date_parts[0] if date_parts else None, doi=item.get("DOI"), url=item.get("URL")))
        return sources
