from types import ModuleType
from typing import Any, Callable, Optional, List, TypeVar

from vision.configs.task import Trainer


M = TypeVar("M", bound=Trainer)

BUILTIN_EXP_CONFIG = {}


def register_experiment_config(
    name: Optional[str] = None,
) -> Callable[[Callable[..., M]], Callable[..., M]]:
    def wrapper(fn: Callable[..., M]) -> Callable[..., M]:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_EXP_CONFIG:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_EXP_CONFIG[key] = fn
        return fn

    return wrapper


def list_config(module: Optional[ModuleType] = None) -> List[str]:
    """
    Returns a list with the names of registered models.

    Args:
        module (ModuleType, optional): The module from
                which we want to extract the available models.

    Returns:
        models (list): A list with the names of available models.
    """
    models = [
        k
        for k, v in BUILTIN_EXP_CONFIG.items()
        if module is None or v.__module__.rsplit(".", 1)[0] == module.__name__
    ]
    return sorted(models)


def get_config_builder(name: str) -> Callable[..., Trainer]:
    """
    Gets the model name and returns the model builder method.

    Args:
        name (str): The name under which the model is registered.

    Returns:
        fn (Callable): The model builder method.
    """
    name = name.lower()
    try:
        fn = BUILTIN_EXP_CONFIG[name]
    except KeyError:
        raise ValueError(f"Unknown exp config {name}")
    return fn


def get_config(dataset_type: str, **config: Any) -> Trainer:
    """
    Gets the model name and configuration and returns an instantiated model.

    Args:
        dataset_type (ModelConfig): config of the model
        **config (Any): parameters passed to the model builder method.

    Returns:
        model (nn.Module): The initialized model.
    """
    fn = get_config_builder(dataset_type)
    return fn(**config)
