"""DeepSeek deployment helper

Provides a simple deployment descriptor for DeepSeek models.
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
    model_name: str = "deepseek-3b", provider: str = "local", **kwargs
) -> DeploymentResult:
    config = {
        "framework": "tensorflow",
        "precision": kwargs.get("precision", "fp32"),
        "device": kwargs.get("device", "cuda" if provider == "local" else "cpu"),
        "optimizations": kwargs.get("optimizations", []),
    }
    config.update(kwargs)
    return DeploymentResult(model=model_name, provider=provider, config=config)


def get_chat_model(
    model_name: str = "deepseek-3b",
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
