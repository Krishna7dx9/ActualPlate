# Backend Runtime Configuration

The backend reads runtime configuration from environment variables. Deployment-specific values must be supplied by the deployment environment or secret manager; do not commit .env files containing credentials.

## Variables

| Variable | Required | Description |
|---|---|---|
| `APP_ENV` | Recommended; set to `production` in production deployments | Runtime environment. Defaults to `development` when unset. |
| `CORS_ALLOWED_ORIGINS` | Required for browser clients hosted on another origin | Comma-separated list of exact trusted origins, including scheme and optional port. An empty value denies cross-origin browser access. Wildcard origins are rejected in production. |
| `VISION_API_TOKEN` | Required when `APP_ENV=production` | Shared service-to-service token used by the backend's vision worker integration. Store it in a secret manager. |

## CORS examples

For local development, configure the exact frontend origin used by the developer:

```text
APP_ENV=development
CORS_ALLOWED_ORIGINS=http://localhost:3000
```

For production, configure the deployed frontend origin:

```text
APP_ENV=production
CORS_ALLOWED_ORIGINS=https://app.example.com
VISION_API_TOKEN=<provided-by-secret-manager>
```

The example host is illustrative; replace it with the actual deployed origin. Do not use the example value as a production configuration.

## Security behavior

- The backend does not enable credentialed CORS.
- The backend CORS middleware owns preflight handling; upload routes do not manually emit wildcard CORS headers.
- An empty origin allowlist is a deny-by-default configuration.
- Production rejects wildcard CORS origins.
- The vision service rejects production startup when VISION_API_TOKEN is missing.
- Public error responses do not include raw exception messages. Detailed diagnostics belong in access-controlled server logs.

## Deployment verification

After setting the variables, verify application startup and test a preflight request from the exact deployed frontend origin. Also verify that an untrusted origin receives no permissive Access-Control-Allow-Origin response header.
