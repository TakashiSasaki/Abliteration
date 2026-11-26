@echo off
setlocal
set LLAMA_SERVER=llama-server.exe
set MODEL=%~dp0..\models\gemma-4-E4B-it-abliterated-5L-FT12-Q4_K_M.gguf
set DRAFT=%~dp0..\models\mtp-assistant-Q8_0.gguf

"%LLAMA_SERVER%" ^
  -m "%MODEL%" ^
  --model-draft "%DRAFT%" ^
  --spec-type draft-mtp ^
  --spec-draft-n-max 4 ^
  -ngl 999 ^
  -ngld 999 ^
  --ctx-size 16384 ^
  -ctk q8_0 ^
  -ctv q8_0 ^
  -fa on ^
  --jinja ^
  --host 127.0.0.1 ^
  --port 8080
