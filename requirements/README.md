# Python dependency manifests

- `backend.txt` is the pinned runtime dependency set for the HTTP backend and shared non-GPU processing code.
- `test.txt` contains test-runner dependencies.
- `vision_service/inference/requirements.txt` contains the inference/model stack. The Docker image pins the CUDA-enabled PyTorch base separately; the MoGe source revision is pinned because the application imports its versioned v2 API.
- The root `requirements.txt` is a convenience aggregate for local development and CI. The production Dockerfile installs `backend.txt` and the inference manifest, not the test tools.

## Updating dependencies

1. Update the appropriate manifest intentionally; do not add packages only to a CI command.
2. Run `python -m pip check` and the test suite.
3. For GPU/model dependencies, validate model loading and a representative inference request on the target CUDA runtime before upgrading pins.
4. Do not claim the Docker image is build-verified until an actual Docker build has completed.
