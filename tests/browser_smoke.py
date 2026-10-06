from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[1]/".agentflow"
with sync_playwright() as p:
    browser=p.chromium.launch(channel="msedge",headless=True)
    page=browser.new_page()
    errors=[]
    page.on("pageerror",lambda e: errors.append(str(e)))
    page.goto("http://127.0.0.1:8001")
    page.locator("#objective").fill("Inspect my paper and dataset")
    page.locator("#files").set_input_files([str(root/"sample-paper.pdf"),str(root/"sample-data.csv")])
    page.get_by_text("sample-data.csv — ready (data)",exact=True).wait_for(timeout=30000)
    page.locator("#start").click()
    page.get_by_role("link",name="Download report").wait_for(timeout=30000)
    assert "Rows: 3" in page.locator("#results").inner_text()
    assert "80 percent" in page.locator("#results").inner_text()
    assert not errors,errors
    page.screenshot(path=str(root/"verified-ui.png"),full_page=True)
    print("BROWSER PASS: multi-file upload -> run -> evidence report -> download link; no JS errors")
    browser.close()

