# Deployment Guide

This guide documents the current repository layout and known deployment prerequisites. The project has active engineering changes in separate pull requests; check the configuration contract in the branch you deploy.

## Services

- Backend API: FastAPI, default port 8000.
- Vision inference API: FastAPI, default port 8001 when started by start.sh.
- The backend calls the vision service through VISION_INFERENCE_ENDPOINT.
- Model weights may be downloaded lazily on first inference, so first-request latency and GPU memory requirements are higher than warm-request requirements.

## Required provider configuration

Supply secrets through environment variables or a secret manager. Never commit them.

- OPENROUTER_API_KEY: credentials for the vision-language recognition provider.
- FATSECRET_CLIENT_ID and FATSECRET_CLIENT_SECRET: credentials for nutrition lookup.
- VISION_INFERENCE_ENDPOINT: the reachable URL of the vision API. For the current combined local startup script, use the loopback endpoint for the worker port.
- VISION_API_TOKEN: configure a shared token for backend-to-vision requests, especially outside local development.
- APP_ENV: set to production for production deployments when using the environment-aware authentication changes.

The vision and backend services must agree on the token and endpoint configuration. Do not expose the vision API publicly unless its authentication and ingress policy are configured.

## Local startup

1. Use Python 3.11 and install the project's backend and inference dependencies according to the requirements files and CI workflow for the branch being tested.
2. Configure provider credentials in your shell or local secret manager.
3. Configure VISION_INFERENCE_ENDPOINT to the local worker address if it is not derived by the checked-out start script.
4. Start both services using the repository's start script.
5. Confirm the backend health endpoint and vision health endpoint respond before submitting an image.

The current main branch's dependency files and startup script are being improved in separate PRs. Until those changes are merged, the Docker build should be treated as unverified; see ENGINEERING_BASELINE_AUDIT.md for the known missing dependency-manifest reference.

## Production requirements

Before production deployment:

- Set APP_ENV=production and provide VISION_API_TOKEN.
- Restrict CORS to the actual browser-client origins.
- Configure explicit request-body, upload-byte, and decoded-image-pixel limits at both the application and ingress/reverse-proxy layers.
- Use a supported CUDA runtime and compatible NVIDIA driver/container runtime when running GPU inference.
- Keep provider credentials in a secret manager and rotate them if exposed.
- Run the test suite, build the container image, and verify startup and worker-to-backend communication.
- Measure latency, memory use, and inference behavior on the actual target hardware.

## Verification status

Repository documentation and CI checks do not prove that a container image has been built successfully or that a GPU inference request works in production. Record those results separately for each deployment environment.
