"""Organize and annotate image artifacts from real cavity LLM repeats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

import cv2

from src.io.artifacts import ensure_dir, write_json
from src.io.schemas import TwoMirrorCavityRealLLMAxisConfig


def organize_and_annotate_repeat(
    repeat_dir: str | Path,
    manifest: dict[str, Any],
    axes: Sequence[TwoMirrorCavityRealLLMAxisConfig],
    *,
    move_existing: bool,
) -> dict[str, Any]:
    """Put capture pairs under images/original and create marked image copies."""
    repeat_path = Path(repeat_dir)
    original_dir = ensure_dir(repeat_path / "images" / "original")
    annotated_dir = ensure_dir(repeat_path / "images" / "annotated")
    path_replacements: dict[str, str] = {}

    if move_existing:
        for image_path in sorted(repeat_path.glob("*.png")):
            destination = original_dir / image_path.name
            old_image = str(image_path)
            image_path.replace(destination)
            path_replacements[old_image] = str(destination)
            metadata_path = image_path.with_suffix(".json")
            if metadata_path.exists():
                metadata_destination = original_dir / metadata_path.name
                old_metadata = str(metadata_path)
                metadata_path.replace(metadata_destination)
                path_replacements[old_metadata] = str(metadata_destination)

    if path_replacements:
        _replace_paths_in_value(manifest, path_replacements)
        _rewrite_messages_json(repeat_path / "messages.json", path_replacements)
        _rewrite_messages_markdown(repeat_path / "messages.md", path_replacements)
        for metadata_path in original_dir.glob("*.json"):
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            _replace_paths_in_value(payload, path_replacements)
            write_json(metadata_path, payload)

    contexts = _capture_contexts(manifest)
    annotations: list[dict[str, Any]] = []
    for image_path in sorted(original_dir.glob("*.png")):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read capture image for annotation: {image_path}")
        context = contexts.get(image_path.name, {})
        physical = context.get("physical_position_steps")
        observation = context.get("llm_observation")
        actual_normalized = _actual_normalized_position(physical, axes)
        llm_normalized = _llm_normalized_position(observation)
        actual_pixel = _normalized_to_pixel(actual_normalized, image.shape[1], image.shape[0])
        llm_pixel = _normalized_to_pixel(llm_normalized, image.shape[1], image.shape[0])

        if actual_pixel is not None:
            cv2.drawMarker(
                image, actual_pixel, (0, 255, 0), cv2.MARKER_TILTED_CROSS,
                markerSize=28, thickness=3, line_type=cv2.LINE_AA,
            )
        if llm_pixel is not None:
            cv2.drawMarker(
                image, llm_pixel, (0, 0, 255), cv2.MARKER_TILTED_CROSS,
                markerSize=18, thickness=2, line_type=cv2.LINE_AA,
            )

        annotated_path = annotated_dir / image_path.name
        if not cv2.imwrite(str(annotated_path), image):
            raise RuntimeError(f"Unable to write annotated image: {annotated_path}")
        annotation = {
            "source_image_path": str(image_path),
            "annotated_image_path": str(annotated_path),
            "physical_position_steps": physical,
            "actual_normalized_position": actual_normalized,
            "actual_pixel": _pixel_payload(actual_pixel),
            "actual_green_x_drawn": actual_pixel is not None,
            "llm_observation": observation,
            "llm_normalized_position": llm_normalized,
            "llm_pixel": _pixel_payload(llm_pixel),
            "llm_red_x_drawn": llm_pixel is not None,
        }
        write_json(annotated_path.with_suffix(".json"), annotation)
        annotations.append(annotation)

    summary = {
        "original_dir": str(original_dir),
        "annotated_dir": str(annotated_dir),
        "image_count": len(annotations),
        "green_x_count": sum(bool(item["actual_green_x_drawn"]) for item in annotations),
        "red_x_count": sum(bool(item["llm_red_x_drawn"]) for item in annotations),
        "legend": {"green_x": "calibration-predicted actual secondary beam", "red_x": "LLM-estimated secondary beam"},
    }
    manifest["image_artifacts"] = summary
    return summary


def process_completed_run(
    run_dir: str | Path,
    axes: Sequence[TwoMirrorCavityRealLLMAxisConfig],
) -> list[dict[str, Any]]:
    """Migrate and annotate every completed repeat in an existing run."""
    results = []
    for repeat_dir in sorted(Path(run_dir).glob("repeat_*")):
        manifest_path = repeat_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = organize_and_annotate_repeat(
            repeat_dir, manifest, axes, move_existing=True
        )
        write_json(manifest_path, manifest)
        results.append({"repeat": repeat_dir.name, **summary})
    return results


def _capture_contexts(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    observations_by_batch = {
        int(decision["input_batch"]): decision["response"].get("image_observations", [])
        for decision in manifest.get("decisions", [])
        if isinstance(decision, dict) and isinstance(decision.get("response"), dict)
    }

    origin_before = manifest.get("physical_origin_before_capture")
    _add_context(contexts, origin_before, {"x": 0, "y": 0}, None)
    initial = manifest.get("initial_capture")
    if isinstance(initial, dict):
        initial_observation = _observation_at(observations_by_batch.get(0), 0)
        _add_context(
            contexts, initial.get("capture"), initial.get("physical_position_steps"), initial_observation
        )
    for batch in manifest.get("batches", []):
        batch_index = int(batch.get("batch_index", 0))
        observations = observations_by_batch.get(batch_index, [])
        for index, capture_item in enumerate(batch.get("captures", [])):
            _add_context(
                contexts,
                capture_item.get("capture"),
                capture_item.get("physical_target_steps"),
                _observation_at(observations, index),
            )
    final = manifest.get("final")
    if isinstance(final, dict):
        _add_context(contexts, final.get("capture"), final.get("physical_position_steps"), None)
    restoration = manifest.get("restoration")
    if isinstance(restoration, dict):
        _add_context(
            contexts, restoration.get("capture"), restoration.get("physical_position_after"), None
        )
    return contexts


def _add_context(
    contexts: dict[str, dict[str, Any]],
    capture: object,
    physical: object,
    observation: object,
) -> None:
    if not isinstance(capture, dict) or not capture.get("llm_image_path"):
        return
    contexts[Path(str(capture["llm_image_path"])).name] = {
        "physical_position_steps": physical,
        "llm_observation": observation,
    }


def _actual_normalized_position(
    physical: object, axes: Sequence[TwoMirrorCavityRealLLMAxisConfig]
) -> dict[str, float] | None:
    if not isinstance(physical, dict) or not all(role in physical for role in ("x", "y")):
        return None
    by_role = {axis.role: axis for axis in axes}
    return {
        role: (
            by_role[role].calibration_offset_normalized
            + by_role[role].calibration_slope_normalized_per_step * int(physical[role])
        )
        for role in ("x", "y")
    }


def _llm_normalized_position(observation: object) -> dict[str, float] | None:
    if not isinstance(observation, dict) or observation.get("visibility") != "visible":
        return None
    if observation.get("x") is None or observation.get("y") is None:
        return None
    return {"x": float(observation["x"]), "y": float(observation["y"])}


def _normalized_to_pixel(
    position: dict[str, float] | None, width: int, height: int
) -> tuple[int, int] | None:
    if position is None or not (-1 <= position["x"] <= 1 and -1 <= position["y"] <= 1):
        return None
    return (
        int(round((position["x"] + 1) * 0.5 * (width - 1))),
        int(round((1 - position["y"]) * 0.5 * (height - 1))),
    )


def _pixel_payload(pixel: tuple[int, int] | None) -> dict[str, int] | None:
    return None if pixel is None else {"x": pixel[0], "y": pixel[1]}


def _observation_at(observations: object, index: int) -> dict[str, Any] | None:
    if not isinstance(observations, list) or index >= len(observations):
        return None
    return observations[index] if isinstance(observations[index], dict) else None


def _replace_paths_in_value(value: object, replacements: dict[str, str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, str) and child in replacements:
                value[key] = replacements[child]
            else:
                _replace_paths_in_value(child, replacements)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, str) and child in replacements:
                value[index] = replacements[child]
            else:
                _replace_paths_in_value(child, replacements)


def _rewrite_messages_json(path: Path, replacements: dict[str, str]) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    _replace_paths_in_value(payload, replacements)
    write_json(path, payload)


def _rewrite_messages_markdown(path: Path, replacements: dict[str, str]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    for old, new in replacements.items():
        if not old.lower().endswith(".png"):
            continue
        text = text.replace(f"({quote(Path(old).name)})", f"(images/original/{quote(Path(new).name)})")
    path.write_text(text, encoding="utf-8")
