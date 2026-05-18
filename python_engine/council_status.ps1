# Poll council status
Invoke-RestMethod -Uri "http://localhost:8001/council/status" -Method GET | ConvertTo-Json -Depth 5
