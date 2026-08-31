from agentflow.research_tools import CrossrefSearchProvider


def test_parses_crossref_items(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b"{}"

    payload = {"message": {"items": [{"title": ["A paper"], "author": [{"given": "Ada", "family": "Lovelace"}], "published": {"date-parts": [[2024]]}, "DOI": "10.1234/example", "URL": "https://doi.org/10.1234/example"}]}}
    monkeypatch.setattr("agentflow.research_tools.urlopen", lambda *args, **kwargs: Response())
    monkeypatch.setattr("agentflow.research_tools.json.load", lambda _: payload)
    sources = CrossrefSearchProvider().search("multi-agent systems")
    assert sources[0].title == "A paper"
    assert sources[0].authors == ["Ada Lovelace"]
    assert sources[0].doi == "10.1234/example"
