# CLI Chatbot (HF Router + Local Fallback)

A lightweight command-line chatbot that first attempts to use Hugging Face's Router (OpenAI-compatible endpoint) with a remote model (`google/gemma-3-27b-it:featherless-ai`) and falls back to a small local model (`distilgpt2`) if remote inference is unavailable. Maintains a sliding window conversation memory.

## Features
- Remote inference via Hugging Face router (OpenAI-compatible API surface)
- Local fallback using `transformers` pipeline
- Sliding window conversational memory (configurable)
- Simple CLI with commands: `/exit`, `/reset`, `/help`
- Modular code (`model_loader.py`, `memory.py`, `interface.py`)

## Setup
1. (Optional) Create & activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Export your Hugging Face token (never hardcode it). On Windows PowerShell:
```powershell
$Env:HF_TOKEN = "YOUR_TOKEN_HERE"
```

## Run
Remote (default attempt):
```powershell
python interface.py
```
Force remote model / override:
```powershell
python interface.py --remote --remote-model google/gemma-3-27b-it:featherless-ai
```
Local only (simulate remote failure by unsetting token):
```powershell
Remove-Item Env:HF_TOKEN
python interface.py
```
Specify local model:
```powershell
python interface.py --local-model distilgpt2
```
Adjust memory window size (turns):
```powershell
python interface.py --window 3
```

## Example Interaction
```
User: What is the capital of France?
Bot: The capital of France is Paris.
User: And what about Italy?
Bot: The capital of Italy is Rome.
User: /exit
Exiting chatbot. Goodbye!
```

## Design Notes
- Remote path uses `openai` client pointed to `https://router.huggingface.co/v1` with model id.
- Local path builds a simple conversational prompt concatenation.
- Memory keeps the last N turns (user+assistant counted as two entries each).
- Commands allow fast iteration and debugging.

## Optional: Jupyter Prototyping
You can create a notebook for quick experiments; not required for main usage.

## Future Improvements
- Add streaming output abstraction for local mode.
- Add colored terminal output via `rich`.
- Persist conversation transcripts.
- Unit tests for memory window edge cases.

## License
MIT (add a LICENSE file if distributing publicly).
