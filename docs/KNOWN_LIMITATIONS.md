# Known Limitations

This document separates implemented behavior from claims that still require independent validation.

## Metric volume and absolute scale

**Absolute volume from one monocular image is not currently validated to a physical accuracy target.**

- A single RGB image does not uniquely reveal absolute scene scale without a metric cue or a learned prior.
- The vision-language model's estimates of food diameter and thickness are uncertain priors, not measured calibration values.
- The MoGe depth output must not be assumed to be meters unless the exact model/configuration and its per-image scale behavior are established.
- The separate ray/plane volume utility requires metric depth, calibrated focal length, and a visible sufficiently planar support surface. Those assumptions do not cover every intended capture condition.
- The current active API path and the standalone geometry utility have historically used different methods.
- Synthetic formula consistency is not evidence of real-world volume accuracy. A measured ground-truth evaluation set is required before claiming the <=5% target.

A null or explicitly unvalidated volume is preferable to a precise-looking value whose physical scale is unsupported. See ADR-0003 for the scale-evidence contract.

## Geometry and segmentation

- Food surfaces may be occluded or hidden.
- Thin, porous, irregular, stacked, or overlapping foods can violate simplified geometric assumptions.
- Detection and segmentation errors propagate into geometry and volume estimates.
- Predicted camera intrinsics are not equivalent to independently calibrated camera intrinsics.

## Nutrition

- Nutrition calculations depend on food identification, volume, density, and provider data quality.
- Any error in physical mass propagates into nutrition values.
- Nutrition data lookup may fail independently of vision processing; callers should distinguish partial results from complete results.

## Operations and reliability

- Model downloads and initialization can make cold starts slow and memory-intensive.
- GPU compatibility depends on the model versions, PyTorch build, CUDA runtime, and host driver.
- External provider calls need bounded timeouts, failure handling, and capacity limits.
- Upload handling must enforce total request size, file size, supported image formats, and decoded pixel limits.
- CI does not run live provider calls, GPU model inference, or physical measurement validation by default.

## Evidence needed to close limitations

1. A documented metric-scale source with provenance and uncertainty.
2. Analytically derived synthetic tests independent of the implementation formula.
3. A held-out real-food dataset with independently measured volume and mass.
4. Error reporting by food geometry, device/camera, distance, lighting, occlusion, and capture condition.
5. Verified failure detection and uncertainty calibration.
6. Successful Docker build and representative deployment tests on target hardware.
