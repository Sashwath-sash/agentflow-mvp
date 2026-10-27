"""Browser-to-API visual-analysis demo check against a supplied paper PDF."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main(path: Path):
    if not path.is_file():
        raise SystemExit("Pass an existing paper PDF path")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8001/", wait_until="domcontentloaded")
        page.locator("#objective").fill("Explain the architecture diagram in this paper")
        page.locator("#files").set_input_files(str(path))
        page.get_by_text(path.name + " — ready (document)", exact=True).wait_for(timeout=30000)
        page.locator("#provider").select_option("none")
        page.locator("#start").click()
        page.get_by_role("link", name="Download report").wait_for(timeout=90000)
        assert "Diagram Analysis Agent" in page.locator("#agents").inner_text()
        assert "Figure 1" in page.locator("#results").inner_text()
        assert page.locator("#results img").count() > 0
        assert not errors, errors
        print("BROWSER PASS: upload -> diagram agent -> OCR labels -> figure preview -> report")
        browser.close()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
