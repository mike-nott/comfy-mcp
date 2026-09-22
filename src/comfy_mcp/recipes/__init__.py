"""Hard-coded ComfyUI API-format graphs, one module per model.

Image recipe modules expose: KEY, NAME, SAVE_NODE, MAX_REFERENCES, ASPECTS, GUIDANCE,
canvas(), text_graph(files, ...), edit_graph(files, ...). Video recipe modules expose:
KEY, NAME, MODES, VIDEO_NODE, POSTER_NODE, REQUIRED_NODES, GUIDANCE, frames(), canvas(),
graph(files, ...). Register new ones here and give them a matching section in the config.
"""

from types import ModuleType

from . import minimax_h3, qwen21

IMAGE_MODELS: dict[str, ModuleType] = {qwen21.KEY: qwen21}
VIDEO_MODELS: dict[str, ModuleType] = {minimax_h3.KEY: minimax_h3}


def image_model(key: str) -> ModuleType:
    try:
        return IMAGE_MODELS[key]
    except KeyError as error:
        raise ValueError(f"Unknown image model '{key}'. Available: {', '.join(IMAGE_MODELS)}") from error


def video_model(key: str) -> ModuleType:
    try:
        return VIDEO_MODELS[key]
    except KeyError as error:
        raise ValueError(f"Unknown video model '{key}'. Available: {', '.join(VIDEO_MODELS)}") from error
