$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python -m uvicorn agentflow.api:app --host 127.0.0.1 --port 8001
