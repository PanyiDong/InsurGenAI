"""
File Name: qwen.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurGenAI
Latest Version: 0.0.1
Relative Path: /InsurGenAI/deployment/qwen.py
File Created: Sunday, 12th October 2025 1:54:05 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 12th October 2025 9:43:38 pm
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

"""Qwen deployment helper

Provides deploy_model(...) which returns a small descriptor for Qwen-based models.
This function is offline-safe (no network) and intended as a convenience wrapper
for constructing deployment metadata used by higher-level runners.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from .unified import get_chat_model


@dataclass
class DeploymentResult:
    model: str
    provider: str
    config: Dict[str, Any]


def deploy_model(
    model_name: str = "Qwen/Qwen3-4b", provider: str = "local", **kwargs
) -> DeploymentResult:
    """Create a deployment descriptor for a Qwen model.

    Inputs:
    - model_name: identifier of the model (e.g., 'Qwen/Qwen3-4b')
    - provider: 'local', 'huggingface', 'azure', etc.
    - kwargs: optional runtime configuration (memory, cpu, batch_size)

    Returns:
    - DeploymentResult dataclass with model, provider, and config dict.
    """
    config = {
        "framework": "pytorch",
        "precision": kwargs.get("precision", "fp16"),
        "device": kwargs.get("device", "cuda" if provider == "local" else "cpu"),
        "batch_size": kwargs.get("batch_size", 8),
    }
    config.update(kwargs)
    return DeploymentResult(model=model_name, provider=provider, config=config)


def get_chat_model(
    model_name: str = "Qwen/Qwen3-4b",
    provider: str = "local",
    device: Optional[int] = None,
    memory: bool = True,
    history_separator: str = "\n",
    max_history: int = 10,
    **kwargs,
):
    """Return (DeploymentResult, chat_callable).

    - device: 0 for the first GPU, -1 for CPU. Default is GPU device 0.
    - memory: whether to enable conversation memory (default True)
    """
    descriptor, chat = get_chat_model(
        model_name,
        provider=provider,
        device=device,
        memory=memory,
        history_separator=history_separator,
        max_history=max_history,
        **kwargs,
    )
    # convert descriptor dict to DeploymentResult dataclass for compatibility
    return (
        deploy_model(
            model_name=model_name, provider=provider, **descriptor.get("config", {})
        ),
        chat,
    )
