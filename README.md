# InsurGenAI

`InsurGenAI` provides a unified, lightweight interface for deploying and chatting with open-source large language models (LLMs).

## Features

- **Unified model loader** — a single `get_chat_model` entry point that
  centralizes defaults (device auto-selection, precision, memory, caching)
  for every supported model family.
- **Per-family deployment helpers** — thin, consistent wrappers for popular
  model families, each with sensible default model IDs:
  - DeepSeek (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`)
  - Gemma (`google/gemma-3-4b-it`)
  - Llama (`meta-llama/Llama-3.2-3B-Instruct`)
  - Mistral (`mistralai/Mistral-7B-v0.3`)
- **Conversation memory** — optional multi-turn history with configurable
  separator and max history length.
- **Hardware aware** — auto-detects CUDA and selects GPU when available,
  falls back to CPU otherwise.
- **HF Hub friendly** — forwards `cache_dir` and `token` to
  `from_pretrained` for controlled model downloads and authenticated access.

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/PanyiDong/InsurGenAI.git
cd InsurGenAI
pip install -e .
```

### Requirements

- Python ≥ 3.11
- PyTorch
- Hugging Face `transformers`

See `requirements.txt` for the full runtime dependency list.

## Quick start

```python
from InsurGenAI.deployment import get_chat_model

# Load a chat model (auto-selects GPU if available, else CPU)
descriptor, chat = get_chat_model(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    memory=True,            # keep conversation history
    max_history=20,         # remember last 20 turns
)

# Single turn
print(chat("Summarize this auto claim: ..."))

# Multi-turn (history is retained when memory=True)
print(chat("What deductible applies?"))
```

## Project structure

```
InsurGenAI/
├── InsurGenAI/
│   ├── __init__.py
│   ├── VERSION
│   └── deployment/
│       ├── __init__.py        # exports get_chat_model
│       ├── unified.py         # centralized loader & defaults
│       ├── hf_utils.py        # HF pipeline wrapper + chat callable
│       ├── deepseek.py        # DeepSeek family helper
│       ├── gemma.py           # Gemma family helper
│       ├── llama.py           # Llama family helper
│       ├── mistral.py         # Mistral family helper
│       ├── phi.py             # Microsoft Phi family helper
│       └── qwen.py            # Qwen family helper
├── example.ipynb              # usage examples
├── pyproject.toml             # build & metadata config
├── setup.py                   # legacy setup script
├── requirements.txt
└── README.md
```

## License

This project is licensed under the [MIT License](LICENSE).
