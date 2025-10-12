"""Gemma deployment helper

Deployment descriptor helper for Gemma family models.
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
    model_name: str = "gemma-1.3b", provider: str = "local", **kwargs
) -> DeploymentResult:
    config = {
        "framework": "pytorch",
        "precision": kwargs.get("precision", "fp16"),
        "device": kwargs.get("device", "cuda" if provider == "local" else "cpu"),
        "sharding": kwargs.get("sharding", False),
    }
    config.update(kwargs)
    return DeploymentResult(model=model_name, provider=provider, config=config)


def get_chat_model(
    model_name: str = "gemma-1.3b",
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
