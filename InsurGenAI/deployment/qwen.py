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
    model_name: str = "qwen-7b", provider: str = "local", **kwargs
) -> DeploymentResult:
    """Create a deployment descriptor for a Qwen model.

    Inputs:
    - model_name: identifier of the model (e.g., 'qwen-7b')
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
    model_name: str = "qwen-7b",
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
