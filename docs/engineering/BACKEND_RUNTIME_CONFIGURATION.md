# Backend Runtime Configuration

The backend reads runtime configuration from environment variables. Deployment-specific values must be supplied by the deployment environment or secret manager; do not commit .env files containing credentials.

## Variables

| Variable | Required | Description |
|---|---|---|
| APP_ENV | No (defaults to development) | Runtime environment. Accepted values are development, test, and production. |
| VISION_INFERENCE_ENDPOINT | Yes | Absolute HTTP(S) URL for the vision worker. The backend fails startup when it is missing or malformed. |
| VISION_API_TOKEN | Required when APP_ENV=production | Shared service-to-service token. Store it in a secret manager. |
| CORS_ALLOWED_ORIGINS | Required for browser clients hosted on another origin | Comma-separated list of exact trusted origins, including scheme and optional port. An empty value denies cross-origin browser access. Wildcard origins are rejected in production. |
| MAX_UPLOAD_BYTES | Yes | Maximum encoded image-file size in bytes. Must be a positive integer. |
| MAX_REQUEST_BODY_BYTES | Yes | Maximum total HTTP request body size, including multipart framing. Must be greater than or equal to MAX_UPLOAD_BYTES. |
| MAX_IMAGE_PIXELS | Yes | Maximum decoded image pixel count (width × height). Must be a positive integer. |
| OPENROUTER_API_KEY | Required for food recognition; required at startup in production | Provider credential for food recognition. Store it in a secret manager. |
| FATSECRET_CLIENT_ID | Required for nutrition lookup | Nutrition-provider client ID. Store it in a secret manager. |
| FATSECRET_CLIENT_SECRET | Required for nutrition lookup | Nutrition-provider client secret. Store it in a secret manager. |

## CORS and local examples

For local development, configure the actual frontend origin and the local vision-worker endpoint:

```text
APP_ENV=development
VISION_INFERENCE_ENDPOINT=http://127.0.0.1:8001
CORS_ALLOWED_ORIGINS=http://localhost:3000
MAX_UPLOAD_BYTES=10485760
MAX_REQUEST_BODY_BYTES=12582912
MAX_IMAGE_PIXELS=40000000
```

These numeric values are examples for local development only. Choose limits based on the deployed service's memory, CPU, inference capacity, and ingress configuration.

For production, set APP_ENV=production, configure the deployed worker endpoint and exact frontend origin, provide VISION_API_TOKEN and OPENROUTER_API_KEY through a secret manager, and set all resource limits explicitly.

## Security behavior

- The backend does not enable credentialed CORS.
- The backend CORS middleware owns preflight handling; upload routes do not manually emit wildcard CORS headers.
- An empty origin allowlist is a deny-by-default configuration.
- Production rejects wildcard CORS origins and fails startup if the vision token or endpoint is missing.
- The ASGI middleware rejects oversized request bodies before multipart parsing. Upload validation separately checks file bytes, supported decoded formats, and pixel count.
- Public error responses do not include raw exception messages. Detailed diagnostics belong in access-controlled server logs.

## Deployment verification

After setting the variables, verify application startup and test a preflight request from the exact deployed frontend origin. Also verify that an untrusted origin receives no permissive Access-Control-Allow-Origin response header. Configure a reverse proxy or ingress request-body limit as defense in depth.
