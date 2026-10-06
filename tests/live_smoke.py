"""Live browser and HTTP smoke checks; creates sample inputs only in ignored storage."""
import io
import time
from pathlib import Path
import requests
import pymupdf
import pandas as pd

BASE="http://127.0.0.1:8001"
root=Path(__file__).resolve().parents[1]/".agentflow"
root.mkdir(exist_ok=True)
pdf=pymupdf.open()
page=pdf.new_page()
page.insert_text((72,72),"Research sample: agents coordinate tools. Accuracy was 80 percent.")
pdf.save(root/"sample-paper.pdf")
pdf.close()
pd.DataFrame({"method":["A","B","C"],"accuracy":[0.8,0.9,None]}).to_csv(root/"sample-data.csv",index=False)
ids=[]
for name in ["sample-paper.pdf","sample-data.csv"]:
    with (root/name).open("rb") as f:
        r=requests.post(BASE+"/upload",files={"file":(name,f)},timeout=30)
    r.raise_for_status()
    ids.append(r.json()["id"])
r=requests.post(BASE+"/execute",json={"objective":"Analyze uploaded paper and dataset","file_ids":ids},timeout=30)
r.raise_for_status()
run_id=r.json()["id"]
for _ in range(60):
    state=requests.get(BASE+"/runs/"+run_id,timeout=10).json()
    if state["status"] not in {"running","queued"}:
        break
    time.sleep(1)
assert state["status"]=="completed",state
report=state["tasks"][-1]["output"]
assert "80 percent" in report["markdown"] and "Rows: 3" in report["markdown"]
assert requests.get(BASE+report["download"],timeout=10).status_code==200
print("LIVE PASS: PDF + CSV -> evidence + statistics + chart + downloadable report")
r=requests.post(BASE+"/research",json={"query":"multi-agent systems","limit":2},timeout=30)
assert r.status_code==200 and r.json()["sources"],r.text
print("LIVE PASS: Crossref scholarly retrieval")
print("Provider configured:",requests.get(BASE+"/health",timeout=10).json()["gemini_configured"])

