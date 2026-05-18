# Full 1M-step council training run — launch AFTER smoke test passes
$body = @{
    symbols            = @("TSLA","NVDA","AAPL","TSM","META","GOOG")
    primary_timeframe  = "1m"
    warmup_steps       = 100000
    total_steps        = 1000000
    eval_every_k_steps = 5000
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8001/council/start" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
