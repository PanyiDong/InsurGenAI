"""
File Name: unified.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurGenAI
Latest Version: 0.0.1
Relative Path: /InsurGenAI/deployment/unified.py
File Created: Sunday, 12th October 2025 2:37:06 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 12th October 2025 3:58:58 pm
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

"""Unified model loader for all models in the deployment package.

Provides a single `get_chat_model` function that centralizes defaults and
loading behavior for different model families. Per-model modules can delegate
to this function so behavior is consistent.
"""
from typing import Tuple, Dict, Any, Optional
from .hf_utils import load_text_generator


def _infer_framework_from_name(model_name: str) -> str:
    name = model_name.lower()
    if "deepseek" in name or "tensorflow" in name:
        return "tensorflow"
    # default to pytorch for most hub models
    return "pytorch"


def get_chat_model(
    model_name: str,
    provider: str = "local",
    device: Optional[int] = None,
    memory: bool = True,
    history_separator: str = "\n",
    max_history: int = 50,
    framework: Optional[str] = None,
    precision: Optional[str] = None,
    batch_size: Optional[int] = None,
    trust_remote_code: bool = False,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    **kwargs,
) -> Tuple[Dict[str, Any], callable]:
    """Centralized loader returning (descriptor_dict, chat_callable).

    - descriptor_dict contains: model, provider, config (a dict)
    - chat_callable(prompt, **generate_kwargs) -> str

    framework and precision can be auto-inferred if not provided.
    """
    framework = framework or _infer_framework_from_name(model_name)
    if precision is None:
        # Default to fp16 for better performance on supported hardware
        precision = "fp16"
    # Auto-select device: prefer GPU if available when device is None
    selected_device = device
    if selected_device is None:
        try:
            import torch

            selected_device = 0 if torch.cuda.is_available() else -1
        except Exception:
            selected_device = -1

    config = {
        "framework": framework,
        "precision": precision,
        # use -1 for CPU, non-negative ints for GPU device ids
        "device": selected_device,
    }
    if batch_size is not None:
        config["batch_size"] = batch_size
    if cache_dir is not None:
        config["cache_dir"] = cache_dir
    if token is not None:
        config["token"] = token
    # merge other kwargs into config for transparency
    cfg = dict(config)
    cfg.update({k: v for k, v in kwargs.items() if k not in {"trust_remote_code"}})

    # load chat callable using hf_utils
    chat = load_text_generator(
        model_name,
        device=selected_device,
        memory=memory,
        history_separator=history_separator,
        max_history=max_history,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        token=token,
        **kwargs,
    )

    descriptor = {"model": model_name, "provider": provider, "config": cfg}
    return descriptor, chat
