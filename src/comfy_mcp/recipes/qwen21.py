"""Qwen Image 2.1 graphs: text-to-image and reference-conditioned editing.

Based on Comfy-Org's native Qwen Image 2.1 workflow template. The only structural
change is the output node: `SaveImageWebsocket` streams PNGs back over the
client's socket instead of writing to ComfyUI's output folder.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Protocol

KEY = "qwen21"
NAME = "Qwen Image 2.1"


class Files(Protocol):
    """The three model files, as ComfyUI's loaders list them (see config `[qwen21]`)."""

    diffusion_model: str
    text_encoder: str
    vae: str

SAVE_NODE = "9"
MAX_REFERENCES = 4
STEP = 32
MIN_EDGE = 256

ASPECTS = {
    "square": (1024, 1024),
    "landscape": (1344, 768),
    "portrait": (768, 1344),
    "wide": (1536, 640),
    "tall": (640, 1536),
}

GUIDANCE = (
    "Qwen Image 2.1 prompt guidance.\n\n"
    "Text to image: describe the subject, setting, lighting, style and medium in plain sentences. "
    "Quoted text renders reliably, e.g. a sign reading \"OPEN\". The negative prompt is optional and "
    "usually unnecessary at cfg 1.0.\n\n"
    "Editing and composing with references: for placing a person or product into another scene, put "
    "the subject image FIRST and the scene image SECOND. Use explicit <image1> and <image2> roles in "
    "the instruction. Write an editing instruction starting with the operation to perform, not only a "
    "description of a finished picture. State what comes from each reference, what must stay "
    "recognizable, and what must change to fulfil the request. The output canvas follows the first "
    "reference unless width/height are given; the first reference is fitted and padded without "
    "stretching, other references keep their own proportions. For a new scene, preserve identity "
    "while allowing pose, lighting and appearance to change as requested; a pasted-looking portrait "
    "is a failed result. For other edits, put the image being changed first. Inspect the result for "
    "missing subjects, identity, proportions and scene integration."
)


@dataclass(frozen=True)
class Reference:
    """An uploaded reference: ComfyUI file token plus its pixel size."""

    token: str  # e.g. "mcp-ab12.png [temp]"
    width: int
    height: int


def round_edge(value: int) -> int:
    return max(MIN_EDGE, int(round(value / STEP)) * STEP)


def canvas(width: int | None, height: int | None, *, aspect: str | None, max_pixels: int, fallback: tuple[int, int] = (1024, 1024)) -> tuple[int, int]:
    """Resolve a requested canvas to multiples of 32 within the pixel budget."""
    if aspect:
        try:
            width, height = ASPECTS[aspect]
        except KeyError as error:
            raise ValueError(f"Unknown aspect '{aspect}'. Choose one of: {', '.join(ASPECTS)}") from error
    if width is None and height is None:
        width, height = fallback
    elif width is None or height is None:
        fw, fh = fallback
        if width is None:
            width = int(round(height * fw / fh))
        else:
            height = int(round(width * fh / fw))
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        width, height = int(width * scale), int(height * scale)
    return round_edge(width), round_edge(height)


def fit_geometry(source_width: int, source_height: int, width: int, height: int) -> tuple[int, int, int, int, int, int]:
    """Fit the whole base reference without stretching/cropping; pad the canvas.

    Returns (fitted_width, fitted_height, left, top, right, bottom).
    """
    scale = min(width / source_width, height / source_height)
    fitted_width = max(1, min(width, round(source_width * scale)))
    fitted_height = max(1, min(height, round(source_height * scale)))
    left = (width - fitted_width) // 2
    top = (height - fitted_height) // 2
    return fitted_width, fitted_height, left, top, width - fitted_width - left, height - fitted_height - top


def _base(files: Files, prompt: str, negative: str, seed: int, steps: int, cfg: float) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": files.diffusion_model, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": files.text_encoder, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": files.vae}},
        "4": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": negative, "resolution": 1024},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["4", 1],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        SAVE_NODE: {"class_type": "SaveImageWebsocket", "inputs": {"images": ["8", 0]}},
    }


def text_graph(files: Files, *, prompt: str, negative: str, width: int, height: int, seed: int, steps: int, cfg: float, count: int) -> dict:
    graph = _base(files, prompt, negative, seed, steps, cfg)
    graph["6"] = {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": count}}
    return graph


def edit_graph(
    files: Files, *, instruction: str, negative: str, references: list[Reference], width: int, height: int, seed: int, steps: int, cfg: float, count: int
) -> dict:
    if not 1 <= len(references) <= MAX_REFERENCES:
        raise ValueError(f"edit_image needs 1 to {MAX_REFERENCES} reference images")
    graph = _base(files, instruction, negative, seed, steps, cfg)
    graph["4"]["inputs"].update(vae=["3", 0], resolution=0)
    graph["6"] = {"class_type": "RepeatLatentBatch", "inputs": {"samples": ["4", 2], "amount": count}}
    for i, ref in enumerate(references):
        load, scale = str(20 + i * 5), str(21 + i * 5)
        graph[load] = {"class_type": "LoadImage", "inputs": {"image": ref.token}}
        if i == 0:
            fitted_w, fitted_h, left, top, right, bottom = fit_geometry(ref.width, ref.height, width, height)
            graph[scale] = {
                "class_type": "ImageScale",
                "inputs": {"image": [load, 0], "upscale_method": "lanczos", "width": fitted_w, "height": fitted_h, "crop": "disabled"},
            }
            graph["22"] = {
                "class_type": "ImagePadForOutpaint",
                "inputs": {"image": [scale, 0], "left": left, "top": top, "right": right, "bottom": bottom, "feathering": 0},
            }
            graph["4"]["inputs"]["images.image_1"] = ["22", 0]
        else:
            graph[scale] = {
                "class_type": "ImageScaleToTotalPixels",
                "inputs": {"image": [load, 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 32},
            }
            graph["4"]["inputs"][f"images.image_{i + 1}"] = [scale, 0]
    return graph
