param(
    [string]$LlamaServer = ".\llama-server.exe",
    [string]$Model = "..\models\gemma-4-E4B-it-abliterated-5L-FT12-Q4_K_M.gguf",
    [string]$DraftModel = "..\models\mtp-assistant-Q8_0.gguf",
    [int]$CtxSize = 16384,
    [int]$DraftNMax = 4,
    [int]$Port = 8080,
    [string]$ListenHost = "127.0.0.1"
)

& $LlamaServer `
  -m $Model `
  --model-draft $DraftModel `
  --spec-type draft-mtp `
  --spec-draft-n-max $DraftNMax `
  -ngl 999 `
  -ngld 999 `
  --ctx-size $CtxSize `
  -ctk q8_0 `
  -ctv q8_0 `
  -fa on `
  --jinja `
  --host $ListenHost `
  --port $Port
