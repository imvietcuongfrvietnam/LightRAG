"""
Frame-semantic extraction using chanind/frame-semantic-transformer.

Replaces LLM-based keyword and entity/relation extraction with neural semantic
frame parsing based on FrameNet lexicography:

  Offline indexing (graph building):
    - Frame Element (FE) texts  →  entity nodes  (entity_type = FE role name)
    - FE pairs within same frame → relation edges (keywords = frame name)

  Online retrieval (query keyword extraction):
    - Frame names               → high_level_keywords (overarching concepts)
    - Frame Element texts       → low_level_keywords  (specific participants)
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from functools import lru_cache

from lightrag.utils import logger
from lightrag.constants import DEFAULT_ENTITY_NAME_MAX_LENGTH


@lru_cache(maxsize=1)
def _get_frame_transformer():
    """Lazy-load and cache the FrameSemanticTransformer singleton (loaded once)."""
    try:
        from frame_semantic_transformer import FrameSemanticTransformer
    except ImportError as exc:
        raise ImportError(
            "frame-semantic-transformer is required. "
            "Install it with: pip install frame-semantic-transformer"
        ) from exc

    logger.info("Loading FrameSemanticTransformer model (first call only) ...")
    transformer = FrameSemanticTransformer()
    logger.info("FrameSemanticTransformer loaded successfully.")
    return transformer


def _detect_frames_sync(text: str):
    """Synchronous frame detection — must run in a thread executor."""
    return _get_frame_transformer().detect_frames(text)


async def detect_frames(text: str):
    """
    Detect semantic frames asynchronously.
    The underlying model is synchronous, so execution is offloaded to a thread.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _detect_frames_sync, text)


# ---------------------------------------------------------------------------
# Online retrieval: query → (hl_keywords, ll_keywords)
# ---------------------------------------------------------------------------

async def extract_keywords_from_frames(
    text: str,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from *text* via frame semantics.

    Returns:
        (hl_keywords, ll_keywords) where
        - hl_keywords: unique detected frame names  (overarching themes)
        - ll_keywords: unique frame element texts   (specific entities/participants)
    """
    try:
        result = await detect_frames(text)
    except Exception as exc:
        logger.error(f"[frame_extractor] Frame detection failed: {exc}")
        return [], []

    seen_frames: set[str] = set()
    seen_fes: set[str] = set()
    hl_keywords: list[str] = []
    ll_keywords: list[str] = []

    for frame in result.frames:
        frame_name = frame.name
        if frame_name and frame_name not in seen_frames:
            hl_keywords.append(frame_name)
            seen_frames.add(frame_name)

        for fe in frame.frame_elements:
            fe_text = fe.text.strip()
            if fe_text and fe_text.lower() not in seen_fes:
                ll_keywords.append(fe_text)
                seen_fes.add(fe_text.lower())

    logger.debug(
        f"[frame_extractor] keywords — hl={hl_keywords} ll={ll_keywords}"
    )
    return hl_keywords, ll_keywords


# ---------------------------------------------------------------------------
# Offline indexing: chunk text → (maybe_nodes, maybe_edges)
# ---------------------------------------------------------------------------

async def extract_entities_from_frames(
    text: str,
    chunk_key: str,
    file_path: str,
) -> tuple[dict, dict]:
    """
    Build graph nodes and edges from *text* via frame-semantic parsing.

    Node mapping
    ~~~~~~~~~~~~
    Each Frame Element (FE) text becomes one entity node:
      - entity_name  = FE text  (truncated to DEFAULT_ENTITY_NAME_MAX_LENGTH)
      - entity_type  = FE role name in lowercase (e.g. "seller", "buyer")
      - description  = auto-generated from frame + role context

    Edge mapping
    ~~~~~~~~~~~~
    Every pair of FEs *within the same frame* becomes one undirected edge:
      - keywords     = frame name (the high-level concept linking them)
      - description  = co-participant description with frame context

    Returns:
        (maybe_nodes, maybe_edges) — same structure as _process_extraction_result()
        maybe_nodes : dict[entity_name,       list[entity_dict]]
        maybe_edges : dict[(src_id, tgt_id),  list[edge_dict]]
    """
    timestamp = int(time.time())

    try:
        result = await detect_frames(text)
    except Exception as exc:
        logger.error(
            f"[frame_extractor] Frame detection failed for chunk {chunk_key}: {exc}"
        )
        return {}, {}

    maybe_nodes: dict[str, list[dict]] = defaultdict(list)
    maybe_edges: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for frame in result.frames:
        frame_name = frame.name
        if not frame_name:
            continue

        # Collect valid FE entities for this frame instance
        fe_list: list[tuple[str, str]] = []  # (entity_name, fe_role)

        for fe in frame.frame_elements:
            fe_text = fe.text.strip()
            if not fe_text:
                continue

            entity_name = fe_text[:DEFAULT_ENTITY_NAME_MAX_LENGTH]
            entity_type = fe.name.lower().replace(" ", "_")
            description = (
                f'"{fe_text}" participates as {fe.name} '
                f'in the "{frame_name}" semantic frame.'
            )

            maybe_nodes[entity_name].append(
                {
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": chunk_key,
                    "file_path": file_path,
                    "timestamp": timestamp,
                }
            )
            fe_list.append((entity_name, fe.name))

        # Create undirected edges between every pair of FEs in this frame
        for i in range(len(fe_list)):
            for j in range(i + 1, len(fe_list)):
                src_name, src_role = fe_list[i]
                tgt_name, tgt_role = fe_list[j]
                if src_name == tgt_name:
                    continue

                description = (
                    f'[{frame_name}] '
                    f'"{src_name}" ({src_role}) '
                    f'co-participates with '
                    f'"{tgt_name}" ({tgt_role}).'
                )
                maybe_edges[(src_name, tgt_name)].append(
                    {
                        "src_id": src_name,
                        "tgt_id": tgt_name,
                        "weight": 1.0,
                        "keywords": frame_name,
                        "description": description,
                        "source_id": chunk_key,
                        "file_path": file_path,
                        "timestamp": timestamp,
                    }
                )

    logger.debug(
        f"[frame_extractor] chunk {chunk_key}: "
        f"{len(maybe_nodes)} nodes, {len(maybe_edges)} edges"
    )
    return dict(maybe_nodes), dict(maybe_edges)
