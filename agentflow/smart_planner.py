from __future__ import annotations

import json

from .local_ai import LocalAIError, OllamaProvider
from .models import TaskGraph
from .planner import Planner


class SmartPlanner:
    """Use a local model for planning when available, with a safe deterministic fallback."""

    def __init__(self, provider: OllamaProvider | None = None, fallback: Planner | None = None) -> None:
        self.provider = provider or OllamaProvider()
        self.fallback = fallback or Planner()
        self.last_mode = "deterministic-fallback"

    def build_graph(self, objective: str) -> TaskGraph:
        objective = objective.strip()
        if not objective:
            raise ValueError("objective must not be empty")
        prompt = (
            "Return only valid JSON matching this schema: "
            '{"objective": "...", "tasks": [{"id": "...", "objective": "...", "role": "...", '
            '"required_capabilities": ["..."], "depends_on": ["..."]}]}. '
            "Create the smallest useful dependency-aware workflow for this objective. "
            f"Objective: {objective}"
        )
        try:
            graph = TaskGraph.model_validate(json.loads(self.provider.generate(prompt)))
            if graph.objective.strip() != objective or not graph.tasks:
                raise ValueError("invalid model graph")
            self.last_mode = "ollama"
            return graph
        except (LocalAIError, ValueError, json.JSONDecodeError, TypeError):
            self.last_mode = "deterministic-fallback"
            return self.fallback.build_graph(objective)
