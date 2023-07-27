from types import ModuleType
from typing import Any, Callable, Optional, List, TypeVar

from torch import nn

from vision.configs import necks as neck_config


M = TypeVar("M", bound=nn.Module)

BUILTIN_MODELS = {}


def register_neck(
    name: Optional[str] = None,
) -> Callable[[Callable[..., M]], Callable[..., M]]:
    def wrapper(fn: Callable[..., M]) -> Callable[..., M]:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_MODELS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_MODELS[key] = fn
        return fn

    return wrapper


def list_model(module: Optional[ModuleType] = None) -> List[str]:
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
        for k, v in BUILTIN_MODELS.items()
        if module is None or v.__module__.rsplit(".", 1)[0] == module.__name__
    ]
    return sorted(models)


def get_neck_builder(name: str) -> Callable[..., nn.Module]:
    """
    Gets the model name and returns the model builder method.

    Args:
        name (str): The name under which the model is registered.

    Returns:
        fn (Callable): The model builder method.
    """
    name = name.lower()
    try:
        fn = BUILTIN_MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name}")
    return fn


def get_neck(
    config: neck_config.Neck,
    in_channels_list: Optional[List[int]] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Gets the model name and configuration and returns an instantiated model.

    Args:
        config (ModelConfig): config of the model
        in_channels_list:
        **kwargs (Any): parameters passed to the model builder method.

    Returns:
        model (nn.Module): The initialized model.
    """
    fn = get_neck_builder(config.type)
    return fn(config, in_channels_list, **kwargs)
