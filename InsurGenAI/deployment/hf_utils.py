"""
File Name: hf_utils.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurGenAI
Latest Version: 0.0.1
Relative Path: /InsurGenAI/deployment/hf_utils.py
File Created: Sunday, 12th October 2025 1:56:50 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 12th October 2025 4:04:05 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2025 - 2025, Panyi Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""Helper utilities to load Hugging Face text-generation/chat models.

Lightweight wrapper that returns a chat(prompt, **generate_kwargs) callable
with optional conversation memory and cache_dir forwarding for model
downloads.
"""
from typing import Callable, Any, List, Tuple, Optional, Union
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


def load_text_generator(
    model_name: str,
    device: int = -1,
    memory: bool = True,
    history_separator: str = "\n",
    max_history: int = 10,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    **kwargs,
) -> Callable[[str], str]:
    """Load a HF causal LM and return a chat callable.

    Parameters:
    - model_name: HF model id or path
    - device: -1 for CPU, >=0 for CUDA device id
    - memory: whether to keep conversation history
    - cache_dir: directory to cache HF model files (forwarded to from_pretrained)
    - kwargs: forwarded to pipeline/from_pretrained (e.g., trust_remote_code)
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, cache_dir=cache_dir, token=token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=kwargs.get("trust_remote_code", False),
        cache_dir=cache_dir,
        token=token,
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        token=token,
        **{k: v for k, v in kwargs.items() if k not in {"trust_remote_code"}},
    )

    history: List[Tuple[str, str]] = []

    def chat(prompt: str, **generate_kwargs: Any) -> Union[str, List[str]]:
        # Build prompt including history if enabled
        if memory and history:
            parts: List[str] = []
            for u, a in history[-max_history:]:
                parts.append(f"User: {u}\nAssistant: {a}")
            prefix = history_separator.join(parts) + history_separator
            prompt_full = prefix + f"User: {prompt}\nAssistant:"
        else:
            prompt_full = f"User: {prompt}\nAssistant:"
        # remove loader-only kwargs so they are not passed to model.generate
        safe_generate_kwargs = {k: v for k, v in generate_kwargs.items()}
        safe_generate_kwargs.pop("cache_dir", None)
        safe_generate_kwargs.pop("trust_remote_code", None)
        safe_generate_kwargs.pop("use_auth_token", None)

        out = gen(prompt_full, **safe_generate_kwargs)

        # If the pipeline returns a list of generation dicts, decode each one.
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            decoded: List[str] = []
            for item in out:
                # prefer common keys used by HF text-generation pipelines
                generated = item.get("generated_text") or item.get("text") or str(item)
                # strip repeated prompt if necessary
                if isinstance(generated, str) and generated.startswith(prompt_full):
                    generated = generated[len(prompt_full) :]
                decoded.append(
                    generated.strip() if isinstance(generated, str) else str(generated)
                )

            # append the first decoded reply to history (if memory enabled)
            if memory and decoded:
                history.append((prompt, decoded[0]))

            # return single string when only one generation; otherwise return list
            return decoded[0] if len(decoded) == 1 else decoded

        # fallback: return stringified output for unknown types
        return str(out)

    def get_history() -> List[Tuple[str, str]]:
        return list(history)

    def clear_history() -> None:
        history.clear()

    setattr(chat, "get_history", get_history)
    setattr(chat, "clear_history", clear_history)
    return chat
