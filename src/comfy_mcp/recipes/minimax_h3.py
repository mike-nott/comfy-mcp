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
# Node ids of optional acceleration patches, in the order they must be chained (Sage before Sol before cache).
SAGE_KJ_NODE, SAGE_H3_NODE, SOL_NODE, FUSED_NODE, CHUNK_NODE, SPECTRUM_NODE, FBC_NODE = "22", "23", "24", "26", "27", "28", "25"
PATCH_NODES = (SAGE_KJ_NODE, SAGE_H3_NODE, SOL_NODE, FUSED_NODE, CHUNK_NODE, SPECTRUM_NODE, FBC_NODE)
UPSCALE_LOADER, UPSCALE_NODE, RESCALE_NODE = "50", "51", "52"


class Accel(Protocol):
    first_block_cache: bool
    fbc_node: str
    fbc_mode: str
    fbc_threshold: float
    fbc_start_percent: float
    fbc_end_percent: float
    fbc_max_consecutive_hits: int
    fbc_temporal_guard: bool
    sol_attn: bool
    sol_node: str
    sol_tau: float
    sol_min_tokens: int
    sol_strict: bool
    sol_thresh_type: str
    sol_int8_qk: bool
    sol_int8_pv: bool
    sol_sink_conditioning: str
    sol_dense_blocks: str
    fused_modulation: bool
    fused_modulation_node: str
    chunk_feed_forward: bool
    chunk_ff_node: str
    chunk_ff_chunks: int
    chunk_ff_min_tokens: int
    sage_patch: bool
    sage_kj_node: str
    sage_kj_mode: str
    sage_h3_node: str
    spectrum: bool
    spectrum_node: str
    spectrum_blend_weight: float
    spectrum_degree: int
    spectrum_warmup_steps: int
    spectrum_tail_actual_steps: int
    spectrum_max_history: int
    draft_width: int
    draft_height: int
    draft_upscale: bool
    upscale_model: str
    final_steps: int
    draft_steps: int


def required_nodes(accel: Accel | None, draft: bool) -> tuple[str, ...]:
    """Base nodes plus whichever acceleration nodes the settings enable for this kind of render."""
    names = list(REQUIRED_NODES)
    if accel:
        if accel.sage_patch:
            names += [n for n in (accel.sage_kj_node, accel.sage_h3_node) if n]
        if accel.sol_attn:
            names.append(accel.sol_node)
        if accel.fused_modulation:
            names.append(accel.fused_modulation_node)
        if accel.chunk_feed_forward:
            names.append(accel.chunk_ff_node)
        if not draft:
            if accel.spectrum:
                names.append(accel.spectrum_node)
            if accel.first_block_cache:
                names.append(accel.fbc_node)
        if draft and accel.draft_upscale:
            names += ["UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScale"]
    return tuple(dict.fromkeys(names))


def accel_summary(accel: Accel | None, draft: bool) -> list[str]:
    if not accel:
        return []
    parts = []
    if accel.sage_patch:
        parts.append("sage")
    if accel.sol_attn:
        parts.append(f"sol-attn tau {accel.sol_tau:g}")
    if accel.fused_modulation:
        parts.append("fused-mod")
    if accel.chunk_feed_forward:
        parts.append("chunk-ff")
    if not draft and accel.spectrum:
        parts.append("spectrum")
    if not draft and accel.first_block_cache:
        parts.append(f"fbc@{accel.fbc_threshold:g}")
    if draft:
        parts.append(f"draft {accel.draft_width}x{accel.draft_height}" + ("+upscale" if accel.draft_upscale else ""))
    return parts

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
    steps: int | None = None,
    accel: Accel | None = None,
) -> tuple[dict, int]:
    """Return (api-format graph, frame count).

    With `accel`, the model passes through an ordered patch chain (Sage -> Sol-Attn -> FirstBlockCache;
    cache on finals only) and the patched model feeds BOTH the scheduler and the guider. Drafts render
    at the accel draft canvas and are upscaled with a SPAN model then rescaled to the final canvas.
    """
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
    render_w, render_h = (width, height)
    if draft and accel:
        render_w, render_h = (accel.draft_width, accel.draft_height) if orientation == "landscape" else (accel.draft_height, accel.draft_width)
    length = frames(seconds)
    step_count = steps or ((accel.draft_steps if accel else DRAFT_STEPS) if draft else (accel.final_steps if accel else FINAL_STEPS))
    model_file = files.reference_model if mode in ("r2v", "refav") else files.diffusion_model

    def node(kind: str, **inputs):
        return {"class_type": kind, "inputs": inputs}

    g = {
        "1": node("UNETLoader", unet_name=model_file, weight_dtype="default"),
        "2": node("CLIPLoader", clip_name=files.text_encoder, type="minimax", device="default"),
        "3": node("VAELoader", vae_name=files.video_vae),
        "4": node("VAELoader", vae_name=files.audio_vae),
        "6": node("MiniMaxH3ImageToVideo", clip=["2", 0], vae=["3", 0], prompt=prompt, width=render_w, height=render_h, length=length),
        "7": node("RandomNoise", noise_seed=seed),
        "8": node("BasicGuider", model=None, conditioning=["6", 0]),
        "9": node("KSamplerSelect", sampler_name="res_multistep"),
        "10": node("BasicScheduler", model=None, scheduler="simple", steps=step_count, denoise=1.0),
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
    # ---- model patch chain: loader -> [turbo LoRA] -> [Sage KJ -> Sage H3] -> [Sol-Attn] -> [FirstBlockCache]
    model_ref = ["1", 0]
    if draft:
        g["20"] = node("MiniMaxH3TurboLoRA", model=model_ref, lora_name=files.turbo_lora, strength=1.0, low_vram=False)
        g["21"] = node("MiniMaxH3TurboSampler")
        model_ref = ["20", 0]
    if accel and accel.spectrum and accel.first_block_cache and not draft:
        raise ValueError("spectrum and first_block_cache must not both be enabled (they conflict on the same model)")
    if accel and accel.sage_patch:
        if accel.sage_kj_node:
            g[SAGE_KJ_NODE] = node(accel.sage_kj_node, model=model_ref, sage_attention=accel.sage_kj_mode)
            model_ref = [SAGE_KJ_NODE, 0]
        g[SAGE_H3_NODE] = node(accel.sage_h3_node, model=model_ref)
        model_ref = [SAGE_H3_NODE, 0]
    if accel and accel.sol_attn:
        g[SOL_NODE] = node(
            accel.sol_node,
            model=model_ref,
            enabled=True,
            tau=accel.sol_tau,
            min_tokens=accel.sol_min_tokens,
            strict=accel.sol_strict,
            thresh_type=accel.sol_thresh_type,
            int8_qk=accel.sol_int8_qk,
            int8_pv=accel.sol_int8_pv,
            sink_conditioning=accel.sol_sink_conditioning,
            dense_blocks=accel.sol_dense_blocks,
        )
        model_ref = [SOL_NODE, 0]
    if accel and accel.fused_modulation:
        g[FUSED_NODE] = node(accel.fused_modulation_node, model=model_ref, enabled=True)
        model_ref = [FUSED_NODE, 0]
    if accel and accel.chunk_feed_forward:
        g[CHUNK_NODE] = node(accel.chunk_ff_node, model=model_ref, enabled=True, chunks=accel.chunk_ff_chunks, min_tokens=accel.chunk_ff_min_tokens)
        model_ref = [CHUNK_NODE, 0]
    if accel and accel.spectrum and not draft:
        g[SPECTRUM_NODE] = node(
            accel.spectrum_node,
            model=model_ref,
            enabled=True,
            blend_weight=accel.spectrum_blend_weight,
            degree=accel.spectrum_degree,
            warmup_steps=accel.spectrum_warmup_steps,
            tail_actual_steps=accel.spectrum_tail_actual_steps,
            max_history=accel.spectrum_max_history,
        )
        model_ref = [SPECTRUM_NODE, 0]
    if accel and accel.first_block_cache and not draft:
        g[FBC_NODE] = node(
            accel.fbc_node,
            model=model_ref,
            mode=accel.fbc_mode,
            threshold=accel.fbc_threshold,
            start_percent=accel.fbc_start_percent,
            end_percent=accel.fbc_end_percent,
            max_consecutive_hits=accel.fbc_max_consecutive_hits,
            temporal_guard=accel.fbc_temporal_guard,
        )
        model_ref = [FBC_NODE, 0]
    g["8"]["inputs"]["model"] = model_ref
    g["10"]["inputs"]["model"] = model_ref
    # ---- draft profile: upscale the small render with SPAN, then rescale to the final canvas
    if draft and accel and accel.draft_upscale and (render_w, render_h) != (width, height):
        g[UPSCALE_LOADER] = node("UpscaleModelLoader", model_name=accel.upscale_model)
        g[UPSCALE_NODE] = node("ImageUpscaleWithModel", upscale_model=[UPSCALE_LOADER, 0], image=["12", 0])
        g[RESCALE_NODE] = node("ImageScale", image=[UPSCALE_NODE, 0], upscale_method="lanczos", width=width, height=height, crop="disabled")
        g[VIDEO_NODE]["inputs"]["images"] = [RESCALE_NODE, 0]
        g["16"]["inputs"]["image"] = [RESCALE_NODE, 0]
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
