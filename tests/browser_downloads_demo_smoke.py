"""Verify the three Downloads demo inputs through the real browser UI."""

from pathlib import Path

from playwright.sync_api import sync_playwright


DOWNLOADS = Path.home() / "Downloads"
INPUTS = [
    DOWNLOADS / "AgentFlow_OCR_Demo.pdf",
    DOWNLOADS / "AgentFlow_Transformer_Paper.pdf",
    DOWNLOADS / "AgentFlow_Demo_8_Models.csv",
]


def main() -> None:
    for path in INPUTS:
        if not path.is_file():
            raise SystemExit(f"Missing demo input: {path}")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8001/", wait_until="domcontentloaded")
        page.locator("#objective").fill(
            "Summarize both papers, inspect the architecture figure, and plot top1 "
            "by model_label from the CSV. The OCR test paper is synthetic."
        )
        page.locator("#files").set_input_files([str(path) for path in INPUTS])
        for path in INPUTS:
            page.get_by_text(path.name, exact=False).first.wait_for(timeout=30000)
        page.locator("#provider").select_option("none")
        page.locator("#start").click()
        page.get_by_role("link", name="Download report").wait_for(timeout=120000)
        results = page.locator("#results").inner_text()
        assert "Pages read with local OCR: 1" in results
        assert "top1 by model_label" in results
        assert "Why other columns were not plotted" in results
        assert "Figure 1" in results
        assert "90.056" in results
        assert page.locator("#results img").count() >= 2
        assert not errors, errors
        print("BROWSER PASS: OCR paper + real research paper + CSV values chart + report")
        browser.close()


if __name__ == "__main__":
    main()
