"""
File Name: mistral.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurGenAI
Latest Version: 0.0.1
Relative Path: /InsurGenAI/deployment/mistral.py
File Created: Sunday, 12th October 2025 1:54:13 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 12th October 2025 9:44:39 pm
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

"""Mistral deployment helper

Light wrapper to create a deployment descriptor for Mistral-family models.
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
    model_name: str = "mistralai/Mistral-7B-v0.3", provider: str = "local", **kwargs
) -> DeploymentResult:
    config = {
        "framework": "pytorch",
        "precision": kwargs.get("precision", "fp16"),
        "device": kwargs.get("device", "cuda" if provider == "local" else "cpu"),
        "max_sequence_length": kwargs.get("max_sequence_length", 2048),
    }
    config.update(kwargs)
    return DeploymentResult(model=model_name, provider=provider, config=config)


def get_chat_model(
    model_name: str = "mistralai/Mistral-7B-v0.3",
    provider: str = "local",
    device: Optional[int] = None,
    memory: bool = True,
    history_separator: str = "\n",
    max_history: int = 10,
    **kwargs,
):
    descriptor, chat = get_chat_model(
        model_name,
        provider=provider,
        device=device,
        memory=memory,
        history_separator=history_separator,
        max_history=max_history,
        **kwargs,
    )
    return (
        deploy_model(
            model_name=model_name, provider=provider, **descriptor.get("config", {})
        ),
        chat,
    )
