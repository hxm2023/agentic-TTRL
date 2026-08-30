#!/usr/bin/env bash
set -e
source /root/miniconda3/bin/activate ttrl2
curl -s -X POST http://localhost:8001/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name":"llama_candidate","lora_path":"/root/autodl-tmp/adapters/llama_candidate"}' \
  | head -c 150
echo
python - << 'PY'
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")
for model in ["llama-3.1-8b", "llama_candidate"]:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say the word: adapter"}],
            max_tokens=30, temperature=0.0)
        print(model, ":", repr(r.choices[0].message.content[:60]))
    except Exception as e:
        print(model, "ERR:", str(e)[:120])
PY
