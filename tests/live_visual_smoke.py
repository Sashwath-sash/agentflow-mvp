"""Inspect local paper figures without modifying saved AgentFlow runs.

Usage: python tests/live_visual_smoke.py path-to-paper.pdf [...]
"""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentflow.visual_tools import has_substantial_figure, inspect_figures
from agentflow import workflow as wf


def main():
    workflow_mode = "--workflow" in sys.argv
    paths = [Path(value) for value in sys.argv[1:] if value != "--workflow"]
    with TemporaryDirectory() as directory:
        for path in paths:
            found = has_substantial_figure(path)
            figures = inspect_figures(path, Path(directory), path.stem, 3) if found else []
            print(f"{path.name}: {len(figures)} figures")
            for figure in figures:
                labels = [item["text"] for item in figure["labels"][:18]]
                print(f'  page {figure["page"]}: {figure["caption"]}')
                print("  labels:", labels)
        if workflow_mode:
            root = Path(directory)
            with patch.object(wf, "ROOT", root), patch.object(wf, "DB", root / "state.sqlite3"):
                files = [wf.add_file(path.name, path.read_bytes()) for path in paths]
                objective = "Compare the uploaded papers and inspect their architecture diagrams"
                run = wf.run_workflow(objective, [file["id"] for file in files])
                print("Workflow:", run["status"], [(task["tool"], task["status"]) for task in run["tasks"]])
                print("Report length:", len(run["tasks"][-1].get("output", {}).get("markdown", "")))
                if run["status"] != "completed":
                    raise SystemExit("Full visual workflow did not complete")


if __name__ == "__main__":
    main()
