"""MiniMax H3 video graphs: text, first/last-frame, reference and reference+audio to video.

Based on Comfy-Org's MiniMax H3 workflow templates. Output goes through
VideoHelperSuite's `VHS_VideoCombine` with save_output=false (ComfyUI temp folder,
fetched then blanked by the client) plus a first-frame poster over the websocket.
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Protocol

KEY = "minimax_h3"
NAME = "MiniMax H3"
MODES = ("t2v", "i2v", "r2v", "refav")
MAX_REFERENCES = 3
FPS = 24
MIN_FRAMES, MAX_FRAMES = 124, 362
FINAL_STEPS, DRAFT_STEPS = 20, 8
VIDEO_NODE = "15"
POSTER_NODE = "17"
REQUIRED_NODES = ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo", "VHS_VideoCombine", "ImageFromBatch", "SaveImageWebsocket")

GUIDANCE = (
    "MiniMax H3 video guidance.\n\n"
    "Describe the subject, the motion and the camera in plain sentences: what moves, how fast, where the camera "
    "is and whether it pans, tracks or holds. One clear action per clip works best; clips are 5-15 seconds. "
    "The model also generates sound, so mention ambient audio or speech if you want it; put spoken lines in "
    "quotes for lip sync. Modes: t2v (text only), i2v (first frame image, optional last frame), r2v (1-3 "
    "reference images of a subject or style, not a starting frame), refav (references plus an audio clip to "
    "follow). draft=true renders an 8-step turbo preview for t2v/i2v only; use it to check composition, then "
    "render the final 20-step version with the same seed."
)


class Files(Protocol):
    diffusion_model: str
    reference_model: str
    text_encoder: str
    video_vae: str
    audio_vae: str
    turbo_lora: str


@dataclass(frozen=True)
class Reference:
    token: str
    width: int
    height: int


def frames(seconds: float) -> int:
    """Snap a duration to H3's 17k+5 frame grid at 24 fps (124-362 frames)."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    return max(MIN_FRAMES, min(MAX_FRAMES, 17 * math.ceil((seconds * FPS - 5) / 17) + 5))


def seconds_for(frame_count: int) -> float:
    return round(frame_count / FPS, 2)


def canvas(orientation: str) -> tuple[int, int]:
    if orientation == "landscape":
        return 1280, 704
    if orientation == "portrait":
        return 704, 1280
    raise ValueError("orientation must be 'landscape' or 'portrait'")


def graph(
    files: Files,
    *,
    prompt: str,
    mode: str,
    images: list[Reference],
    audio_token: str | None,
    seconds: float,
    orientation: str,
    draft: bool,
    seed: int,
) -> tuple[dict, int]:
    """Return (api-format graph, frame count)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    if draft and mode not in ("t2v", "i2v"):
        raise ValueError("draft (turbo) rendering supports t2v and i2v only")
    if mode == "t2v" and images:
        raise ValueError("t2v takes no images; use i2v for a starting frame or r2v for references")
    if mode == "i2v" and not 1 <= len(images) <= 2:
        raise ValueError("i2v needs one image (first frame) or two (first and last frame)")
    if mode in ("r2v", "refav") and not 1 <= len(images) <= MAX_REFERENCES:
        raise ValueError(f"{mode} needs 1 to {MAX_REFERENCES} reference images")
    if mode == "refav" and not audio_token:
        raise ValueError("refav needs an audio clip")
    if audio_token and mode != "refav":
        raise ValueError("audio is only used by refav")

    width, height = canvas(orientation)
    length = frames(seconds)
    model_file = files.reference_model if mode in ("r2v", "refav") else files.diffusion_model
    model_ref = ["20", 0] if draft else ["1", 0]

    def node(kind: str, **inputs):
        return {"class_type": kind, "inputs": inputs}

    g = {
        "1": node("UNETLoader", unet_name=model_file, weight_dtype="default"),
        "2": node("CLIPLoader", clip_name=files.text_encoder, type="minimax", device="default"),
        "3": node("VAELoader", vae_name=files.video_vae),
        "4": node("VAELoader", vae_name=files.audio_vae),
        "6": node("MiniMaxH3ImageToVideo", clip=["2", 0], vae=["3", 0], prompt=prompt, width=width, height=height, length=length),
        "7": node("RandomNoise", noise_seed=seed),
        "8": node("BasicGuider", model=model_ref, conditioning=["6", 0]),
        "9": node("KSamplerSelect", sampler_name="res_multistep"),
        "10": node("BasicScheduler", model=model_ref, scheduler="simple", steps=DRAFT_STEPS if draft else FINAL_STEPS, denoise=1.0),
        "11": node(
            "SamplerCustomAdvanced",
            noise=["7", 0],
            guider=["8", 0],
            sampler=["21" if draft else "9", 0],
            sigmas=["10", 0],
            latent_image=["6", 1],
        ),
        "12": node("VAEDecode", samples=["11", 0], vae=["3", 0]),
        "13": node("VAEDecodeAudio", samples=["11", 0], vae=["4", 0]),
        VIDEO_NODE: node(
            "VHS_VideoCombine",
            images=["12", 0],
            audio=["13", 0],
            frame_rate=float(FPS),
            loop_count=0,
            filename_prefix=f"mcp-{secrets.token_hex(8)}",
            format="video/h264-mp4",
            pix_fmt="yuv420p",
            crf=19,
            save_metadata=False,
            trim_to_audio=False,
            pingpong=False,
            save_output=False,
        ),
        "16": node("ImageFromBatch", image=["12", 0], batch_index=0, length=1),
        POSTER_NODE: node("SaveImageWebsocket", images=["16", 0]),
    }
    if draft:
        g["20"] = node("MiniMaxH3TurboLoRA", model=["1", 0], lora_name=files.turbo_lora, strength=1.0, low_vram=False)
        g["21"] = node("MiniMaxH3TurboSampler")
    if mode in ("r2v", "refav"):
        g["6"]["class_type"] = "MiniMaxH3ReferenceToVideo"
        g["6"]["inputs"].update(audio_vae=["4", 0], ref_image_size="match")
    for i, ref in enumerate(images):
        key = str(30 + i)
        g[key] = node("LoadImage", image=ref.token)
        if mode == "i2v":
            g["6"]["inputs"]["first_frame" if i == 0 else "last_frame"] = [key, 0]
        else:
            g["6"]["inputs"][f"ref_images.ref_image_{i}"] = [key, 0]
    if audio_token:
        g["40"] = node("LoadAudio", audio=audio_token)
        g["6"]["inputs"]["ref_audios.ref_audio_0"] = ["40", 0]
    return g, length
