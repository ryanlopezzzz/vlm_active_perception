"""Extract empirical two-mirror cavity beam profiles from lab camera images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "src/sim/assets/two_mirror_cavity_profiles.npz"

FRAME_WIDTH_PX = 960
FRAME_HEIGHT_PX = 540
BACKGROUND_PERCENTILE = 1.0
SECONDARY_CENTER_NORMALIZED = (-0.77, -0.08)
SECONDARY_MASK_RADII_PX = (105.0, 90.0)
SECONDARY_MASK_INNER_RADIUS = 0.85
SECONDARY_MASK_OUTER_RADIUS = 1.15
MAIN_CENTER_NORMALIZED = (-0.02209929523809521, -0.022140175)
X_SCREEN_NORMALIZED_PER_STEP = -0.00020900476190476185
Y_SCREEN_NORMALIZED_PER_STEP = 0.0003231375000000002


def _read_grayscale(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    if image.shape != (FRAME_HEIGHT_PX, FRAME_WIDTH_PX):
        raise ValueError(
            f"Expected a {FRAME_WIDTH_PX}x{FRAME_HEIGHT_PX} image, got "
            f"{image.shape[1]}x{image.shape[0]} from {path}."
        )
    return image


def _normalized_to_pixel(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return 0.5 * width * (x + 1.0), 0.5 * height * (1.0 - y)


def _soft_ellipse(
    shape: tuple[int, int],
    center_x_px: float,
    center_y_px: float,
    radius_x_px: float,
    radius_y_px: float,
) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float64)
    radius = np.sqrt(
        ((x - center_x_px) / radius_x_px) ** 2
        + ((y - center_y_px) / radius_y_px) ** 2
    )
    weight = np.clip(
        (SECONDARY_MASK_OUTER_RADIUS - radius)
        / (SECONDARY_MASK_OUTER_RADIUS - SECONDARY_MASK_INNER_RADIUS),
        0.0,
        1.0,
    )
    return weight * weight * (3.0 - 2.0 * weight)


def _phase_correlation_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float]:
    """Return the integer-pixel (dy, dx) shift that aligns moving to reference."""
    height, width = reference.shape
    crop_y = slice(height // 2 - 190, height // 2 + 190)
    crop_x = slice(width // 2 - 230, width // 2 + 230)
    fixed = reference[crop_y, crop_x].astype(np.float64)
    candidate = moving[crop_y, crop_x].astype(np.float64)
    window = np.outer(np.hanning(fixed.shape[0]), np.hanning(fixed.shape[1]))
    fixed = (fixed - np.mean(fixed)) * window
    candidate = (candidate - np.mean(candidate)) * window
    cross_power = np.fft.fft2(fixed) * np.conj(np.fft.fft2(candidate))
    cross_power /= np.maximum(np.abs(cross_power), np.finfo(np.float64).eps)
    correlation = np.abs(np.fft.ifft2(cross_power))
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    if peak_y > correlation.shape[0] // 2:
        peak_y -= correlation.shape[0]
    if peak_x > correlation.shape[1] // 2:
        peak_x -= correlation.shape[1]
    return float(peak_y), float(peak_x)


def _weighted_centroid(image: np.ndarray) -> tuple[float, float]:
    total = float(np.sum(image))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("The extracted secondary profile has no positive intensity.")
    y, x = np.indices(image.shape, dtype=np.float64)
    return float(np.sum(x * image) / total), float(np.sum(y * image) / total)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_capture_metadata(image_path: Path) -> dict[str, Any]:
    metadata_path = image_path.with_suffix(".json")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def extract_profiles(
    main_image_path: str | Path,
    combined_image_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract centered, sensor-scaled main and secondary intensity profiles."""
    main_path = Path(main_image_path).resolve()
    combined_path = Path(combined_image_path).resolve()
    main_raw = _read_grayscale(main_path)
    combined_raw = _read_grayscale(combined_path)

    main_floor = float(np.percentile(main_raw, BACKGROUND_PERCENTILE))
    combined_floor = float(np.percentile(combined_raw, BACKGROUND_PERCENTILE))
    main = np.maximum(main_raw - main_floor, 0.0)
    combined = np.maximum(combined_raw - combined_floor, 0.0)

    alignment_shift_y, alignment_shift_x = _phase_correlation_shift(main, combined)
    combined = ndimage.shift(
        combined,
        shift=(alignment_shift_y, alignment_shift_x),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    combined_raw_aligned = ndimage.shift(
        combined_raw,
        shift=(alignment_shift_y, alignment_shift_x),
        order=1,
        mode="constant",
        cval=combined_floor,
        prefilter=False,
    )

    source_center_x, source_center_y = _normalized_to_pixel(
        *SECONDARY_CENTER_NORMALIZED,
        FRAME_WIDTH_PX,
        FRAME_HEIGHT_PX,
    )
    source_center_x += alignment_shift_x
    source_center_y += alignment_shift_y
    source_mask = _soft_ellipse(
        main.shape,
        source_center_x,
        source_center_y,
        *SECONDARY_MASK_RADII_PX,
    )

    fit_pixels = (
        (source_mask == 0.0)
        & (main_raw < 245.0)
        & (combined_raw_aligned < 245.0)
        & (main_raw > main_floor)
    )
    design = np.column_stack((main[fit_pixels], np.ones(int(np.sum(fit_pixels)))))
    gain, offset = np.linalg.lstsq(design, combined[fit_pixels], rcond=None)[0]
    residual = np.maximum(combined - (gain * main + offset), 0.0)
    secondary_source = residual * source_mask

    extracted_center_x, extracted_center_y = _weighted_centroid(secondary_source)
    main_center_x, main_center_y = _normalized_to_pixel(
        *MAIN_CENTER_NORMALIZED,
        FRAME_WIDTH_PX,
        FRAME_HEIGHT_PX,
    )
    recenter_shift = (
        main_center_y - extracted_center_y,
        main_center_x - extracted_center_x,
    )
    secondary_centered = ndimage.shift(
        secondary_source,
        shift=recenter_shift,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    support_centered = ndimage.shift(
        source_mask,
        shift=recenter_shift,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    sensor_range = 255.0
    # Preserve the main camera frame exactly. Floor-subtracted copies above are
    # used only to register the frames and isolate the secondary contribution.
    main_intensity = (main_raw / sensor_range).astype(np.float32)
    secondary_intensity = np.maximum(secondary_centered / sensor_range, 0.0).astype(np.float32)
    secondary_support = np.clip(support_centered, 0.0, 1.0).astype(np.float32)

    main_metadata = _load_capture_metadata(main_path)
    combined_metadata = _load_capture_metadata(combined_path)
    comparable_camera_fields = ("llm_image_shape", "average_frames", "exposure_ms", "analog_gain")
    mismatches = {
        key: [main_metadata.get(key), combined_metadata.get(key)]
        for key in comparable_camera_fields
        if main_metadata.get(key) != combined_metadata.get(key)
    }
    if mismatches:
        raise ValueError(f"Calibration images have incompatible camera settings: {mismatches}")

    arrays = {
        "main_intensity": main_intensity,
        "secondary_intensity": secondary_intensity,
        "secondary_support": secondary_support,
        "main_center_px": np.asarray([main_center_x, main_center_y], dtype=np.float64),
    }
    provenance: dict[str, Any] = {
        "description": "Empirical two-mirror cavity profiles extracted from final real-lab runs.",
        "main_source": {
            "path": _provenance_path(main_path),
            "sha256": _sha256(main_path),
            "capture_metadata": main_metadata,
        },
        "combined_source": {
            "path": _provenance_path(combined_path),
            "sha256": _sha256(combined_path),
            "capture_metadata": combined_metadata,
        },
        "frame_shape": [FRAME_HEIGHT_PX, FRAME_WIDTH_PX],
        "background_percentile": BACKGROUND_PERCENTILE,
        "background_levels_uint8": {"main": main_floor, "combined": combined_floor},
        "combined_to_main_alignment_shift_px": {
            "x": alignment_shift_x,
            "y": alignment_shift_y,
        },
        "photometric_fit": {"gain": float(gain), "offset": float(offset)},
        "secondary_source_center_normalized": list(SECONDARY_CENTER_NORMALIZED),
        "secondary_mask_radii_px": list(SECONDARY_MASK_RADII_PX),
        "secondary_mask_taper": {
            "inner_radius": SECONDARY_MASK_INNER_RADIUS,
            "outer_radius": SECONDARY_MASK_OUTER_RADIUS,
        },
        "extracted_secondary_center_px": [extracted_center_x, extracted_center_y],
        "main_center_normalized": list(MAIN_CENTER_NORMALIZED),
        "main_center_px": [main_center_x, main_center_y],
        "secondary_recenter_shift_px": {"x": recenter_shift[1], "y": recenter_shift[0]},
        "main_profile_processing": "Unmodified 8-bit camera image divided by 255.",
        "secondary_extraction_background_percentile": BACKGROUND_PERCENTILE,
        "normalization_sensor_range_uint8": sensor_range,
        "profile_statistics": {
            "main_peak": float(np.max(main_intensity)),
            "secondary_peak": float(np.max(secondary_intensity)),
            "main_sum": float(np.sum(main_intensity)),
            "secondary_sum": float(np.sum(secondary_intensity)),
        },
        "steering_calibration": {
            "x_screen_normalized_per_step": X_SCREEN_NORMALIZED_PER_STEP,
            "y_screen_normalized_per_step": Y_SCREEN_NORMALIZED_PER_STEP,
            "x_image_pixels_per_step": X_SCREEN_NORMALIZED_PER_STEP * FRAME_WIDTH_PX / 2.0,
            "y_image_pixels_per_step": -Y_SCREEN_NORMALIZED_PER_STEP * FRAME_HEIGHT_PX / 2.0,
            "x_step_limits": [-8000, 8000],
            "y_step_limits": [-5000, 5000],
        },
        "limitations": [
            "The source camera images are 8-bit and saturated in the main-beam core.",
            "The secondary profile is a localized positive intensity difference, not a recovered complex field.",
        ],
    }
    return arrays, provenance


def write_profiles(
    main_image_path: str | Path,
    combined_image_path: str | Path,
    output_path: str | Path,
) -> tuple[Path, Path]:
    arrays, provenance = extract_profiles(main_image_path, combined_image_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    provenance_path = output.with_suffix(".json")
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return output, provenance_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-image", type=Path, required=True)
    parser.add_argument("--combined-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_path, provenance_path = write_profiles(
        args.main_image,
        args.combined_image,
        args.output,
    )
    print(f"Wrote profiles: {profile_path}")
    print(f"Wrote provenance: {provenance_path}")


if __name__ == "__main__":
    main()
