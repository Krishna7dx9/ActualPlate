# System health, readiness, and GPU warmup

## Backend endpoints

- `GET /v1/system/health`: lightweight public health report for database, rate-limit store, and the vision worker's reported model state. It does not expose credentials or raw exceptions.
- `GET /v1/system/ready`: returns 200 only when the database and rate-limit store are available and the vision worker reports all configured models loaded; otherwise it returns 503.
- `POST /v1/system/warmup`: privileged owner/admin action requiring the `system:warmup` scope. It asks the internal vision worker to load Grounding DINO, SAM2, and MoGe models.

Use `/health` for liveness and `/ready` for traffic readiness. A cold worker may be alive but not ready. Do not configure an orchestrator to repeatedly restart a healthy cold worker solely because readiness is not yet true.

## Worker behavior

- `POST /warmup` on the internal vision service requires `X-API-Token`.
- Model loading is executed in a worker thread and guarded by a process-wide lock so concurrent warmup/first inference calls do not initialize duplicate model instances.
- The warmup timeout is configured through `VISION_WARMUP_TIMEOUT_SECONDS`, accepted range 30–900 seconds, default 300.
- The warmup endpoint covers the vision worker's Grounding DINO, SAM2, and MoGe models. It does not validate FatSecret credentials, email delivery, or the full B2B request path.

## Production API docs and headers

- Backend and vision-worker `/docs`, `/redoc`, and `/openapi.json` are disabled when `APP_ENV=production`.
- The backend applies `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, and `Cross-Origin-Opener-Policy`.
- Production API responses receive a restrictive CSP. HSTS is emitted only when the ASGI request scheme is HTTPS; deployments behind a TLS-terminating proxy must configure trusted proxy scheme handling at the edge.
- Health and readiness probes bypass request rate limiting so they can report the state of the rate-limit backend itself. They still receive security headers.

## Operational requirements

- Keep the worker port private to the backend network; do not expose `/warmup` publicly.
- Configure the same strong `VISION_API_TOKEN` in backend and worker secret storage.
- Use an orchestrator/job or a protected operator action for cold-start warmup; do not warm models on every user scan.
- Monitor warmup duration, model-load failures, GPU memory, and worker readiness. The endpoint does not prove inference accuracy or physical-volume validity.
- Production PostgreSQL/Redis deployment, secret rotation, alert routing, and load tests remain release gates.
