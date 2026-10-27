"""Check automatic table discovery through the upload form and rendered page."""

from playwright.sync_api import sync_playwright


DATA = (
    b"date,revenue,conversion_rate,segment\n"
    b"2026-01-01,10,0.1,A\n2026-01-01,20,0.2,B\n"
    b"2026-02-01,30,0.3,A\n2026-03-01,40,0.4,B\n"
    b"2026-04-01,50,,A\n2026-05-01,60,0.5,B\n"
)


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8001/", wait_until="domcontentloaded")
        page.locator("#objective").fill("Discover trends and relationships in this dataset")
        page.locator("#files").set_input_files({
            "name": "dated-demo.csv", "mimeType": "text/csv", "buffer": DATA,
        })
        page.get_by_text("dated-demo.csv", exact=False).first.wait_for(timeout=30000)
        page.locator("#provider").select_option("none")
        page.locator("#start").click()
        page.get_by_role("link", name="Download report").wait_for(timeout=90000)
        text = page.locator("#results").inner_text()
        assert "Time trend: revenue over date" in text
        assert "Time trend: conversion_rate over date" in text
        assert "Relationship between" in text
        assert "Missing-value counts" in text
        assert "Why other columns were not plotted" in text
        assert page.locator("#results img").count() == 4
        assert not errors, errors
        print("BROWSER PASS: dated CSV -> two trends + relationship + missingness + report")
        browser.close()


if __name__ == "__main__":
    main()
