# Inference Pipeline

This document describes the current implemented vision path and distinguishes model outputs from physically validated measurements.

## Stage 1: Food Recognition

- **Input:** Uploaded image bytes.
- **Processing:** The configured vision-language provider identifies food labels and may return visual estimates for density, diameter, and thickness.
- **Output:** Food candidates and optional dimension estimates.
- **Important:** A language-model estimate of physical dimensions is a prior/guess, not a metric calibration measurement. Missing dimensions remain null; no typical-size defaults are inserted.

## Stage 2: Object Detection

- **Input:** Image plus food-label prompts.
- **Processing:** Grounding DINO predicts bounding boxes and label scores.
- **Output:** 2D bounding boxes and labels.

## Stage 3: Instance Segmentation

- **Input:** Image plus detection boxes.
- **Processing:** SAM2 predicts instance masks.
- **Output:** Per-food binary masks.

## Stage 4: Monocular Geometry

- **Input:** RGB image.
- **Processing:** MoGe-2 predicts monocular geometry/depth and camera intrinsics.
- **Output:** A model-relative depth map and predicted focal length.
- **Units:** The current inference contract labels the depth output as relative, not meters. A predicted focal length is not itself evidence of absolute scene scale.

## Stage 5: Metric Volume

A physical volume in cm³ requires a valid metric scale. The current single-image pipeline does not establish that scale independently. VLM-estimated diameter/thickness values may be used for experimental diagnostics only.

The API therefore:
- returns volume_cm3 = null when metric scale evidence is absent;
- reports volume_status = unvalidated when the experimental calculation uses VLM-estimated dimensions;
- exposes any diagnostic result under experimental_volume_cm3, never as the authoritative physical volume;
- does not allow the backend to convert an unvalidated diagnostic volume into food mass or nutrition.

The separate ray/plane volume utility requires metric depth in meters, calibrated focal length, and a sufficiently visible planar supporting surface. It is not a general solution for captures without a visible planar support and is not the active production volume path.

## Stage 6: Nutrition Enrichment

Nutrition enrichment requires a usable physical mass derived from validated volume and density. When no validated volume is available, nutrition enrichment is reported as unavailable rather than silently calculating nutrition from an unvalidated scale estimate.

## Metric-scale requirement

Absolute scale is not uniquely observable from a single monocular image without an additional metric cue or a learned prior. See ADR-0003-metric-volume-requires-scale-evidence.md for the current engineering decision and acceptance criteria.
