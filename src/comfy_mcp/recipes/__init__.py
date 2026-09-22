"""Hard-coded ComfyUI API-format graphs, one module per model.

Each image recipe module exposes: KEY, NAME, SAVE_NODE, MAX_REFERENCES, ASPECTS, GUIDANCE,
canvas(), text_graph(files, ...), edit_graph(files, ...). Register new ones in IMAGE_MODELS and
give them a matching section in the config file.
"""

from types import ModuleType

from . import qwen21

IMAGE_MODELS: dict[str, ModuleType] = {qwen21.KEY: qwen21}


def image_model(key: str) -> ModuleType:
    try:
        return IMAGE_MODELS[key]
    except KeyError as error:
        raise ValueError(f"Unknown image model '{key}'. Available: {', '.join(IMAGE_MODELS)}") from error
