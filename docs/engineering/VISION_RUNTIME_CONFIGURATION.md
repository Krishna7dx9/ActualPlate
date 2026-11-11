# Vision Runtime Configuration

The vision service loads its validated settings from the process environment and, for local development, a non-secret .env file. Copy .env.example to .env and replace the credential placeholders. Never commit .env.

## Required configuration

| Variable | Required | Purpose |
|---|---|---|
| APP_ENV | Yes | Runtime environment. Production enables fail-closed credential checks. |
| OPENROUTER_BASE_URL | Yes | Absolute HTTP(S) provider endpoint. |
| OPENROUTER_MODEL | Yes | Model identifier for food recognition. |
| OPENROUTER_MAX_TOKENS | Yes | Positive integer response token budget. |
| GROUNDING_DINO_MODEL_ID | Yes | Model identifier for open-vocabulary detection. |
| SAM2_MODEL_ID | Yes | Model identifier for segmentation. |
| MOGE_MODEL_ID | Yes | Model identifier for monocular geometry. |
| BOX_THRESHOLD | Yes | Detection confidence threshold in [0, 1]. |
| TEXT_THRESHOLD | Yes | Text matching threshold in [0, 1]. |
| SHOW_VISUALIZATIONS | Yes | Boolean switch for optional visualizations. |
| VISION_MAX_ARRAY_SIDE | Yes | Positive maximum side length for optional array payloads. |
| VISION_REQUEST_TIMEOUT_SECONDS | Yes | Positive finite timeout for backend-to-vision requests. |

## Conditional configuration

- OPENROUTER_API_KEY is required when food recognition is used and required at startup in production.
- VISION_API_TOKEN is optional for local development but required when APP_ENV=production.
- VISION_INFERENCE_ENDPOINT is required by the backend worker client. For a combined local process, configure the reachable worker endpoint (typically the loopback address and worker port). The endpoint must be an absolute HTTP(S) URL.
- VISION_LABEL_ALIASES_FILE optionally points to an alternate JSON alias file. If unset, the service loads the versioned package asset at vision_service/recognition/label_aliases.json.

## Validation behavior

- Numeric values are parsed and validated at startup; invalid values fail startup instead of silently falling back.
- Model identifiers and thresholds are centralized in one settings object.
- Provider clients are initialized lazily. Missing credentials produce a clear error at the operation boundary; production requires the provider credential at startup.
- Secrets are not logged.
- Label aliases and the food-recognition prompt are versioned data assets, separate from Python control flow.

## Security and operations

Use a secret manager for provider tokens in production. Keep the vision endpoint on a private network or behind authenticated ingress. Set request/upload limits in the backend and at the reverse proxy. Model IDs and thresholds should be changed only with an evaluation plan and regression tests.
