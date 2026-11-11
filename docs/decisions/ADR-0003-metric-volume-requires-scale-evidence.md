# ADR-0003: Require Independent Metric-Scale Evidence for Physical Volume

## Status
Accepted

## Context
The current vision-language model estimates food diameter and thickness from a single RGB image. The API previously combined those estimates with a normalized monocular depth map and marked the resulting cm³ value as valid. That label overstated what the inputs establish: VLM size estimates are uncertain priors, and a relative depth map does not independently provide absolute scene scale.

For a pinhole camera, image coordinates depend on ratios such as X/Z and Y/Z. Scaling every scene coordinate by the same positive factor leaves the projection unchanged. A single monocular image therefore does not uniquely determine absolute object dimensions without an additional metric cue or a learned prior. A learned prior may be useful statistically, but it is not a per-image measurement guarantee.

## Decision
- Treat MoGe-2 depth as model-relative unless the exact model/configuration and evaluation establish a metric scale contract.
- Treat VLM diameter/thickness as unvalidated visual estimates, never as metric calibration evidence.
- Do not populate the authoritative volume_cm3 field without independent metric-scale evidence.
- If a diagnostic calculation is retained, expose it only as experimental_volume_cm3 with volume_status = unvalidated.
- Remove fabricated default dimensions. Missing or invalid dimension estimates remain null.
- Do not derive customer-facing mass or nutrition from unvalidated volume.
- Remove any focal-length fallback that substitutes an arbitrary value when camera intrinsics are unavailable. Require a valid predicted or externally calibrated focal value for geometric operations that need it.

## Metric-scale evidence contract
A future production volume path must accept a typed scale-evidence record containing, at minimum:
- source/method identifier and version;
- metric quantity and unit;
- calibration provenance and relevant device/capture metadata;
- uncertainty or an explicitly defined confidence representation;
- validation status and applicable operating conditions.

The scale source must be independently evaluated against measured ground truth. A UI overlay or virtual box is not scale evidence by itself unless its dimensions are tied to a validated metric measurement process.

## Consequences
- The API may return null authoritative volume while the scale problem remains unsolved. This is preferable to presenting a precise-looking but unsupported physical measurement.
- Existing consumers must distinguish experimental estimates from validated measurements.
- Nutrition enrichment may be unavailable until a metric scale source is validated.
- The plane-intersection utility remains conditional on metric depth, calibrated intrinsics, and visible planar support; it is not a universal solution for the product's capture requirements.

## Acceptance criteria for a future metric volume implementation
1. The metric-scale source has a documented, reproducible measurement model.
2. Evaluation uses independently measured real food samples and held-out capture conditions.
3. Error metrics are reported by food geometry, camera/device, lighting, distance, and occlusion where applicable.
4. Uncertainty and failure detection are calibrated on held-out data.
5. The target error (including the desired <=5% threshold) is demonstrated on a predeclared test protocol, not inferred from synthetic formula consistency.
