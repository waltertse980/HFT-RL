# Smoke test — 30k steps, eval every 1k
$body = @{
    symbols            = @("TSLA","NVDA","AAPL","TSM","META","GOOG")
    primary_timeframe  = "1m"
    warmup_steps       = 10000
    total_steps        = 30000
    eval_every_k_steps = 1000
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8001/council/start" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
