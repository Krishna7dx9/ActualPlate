# Rate limiting, daily quotas, and usage accounting

## Two separate controls

ActualPlate now has two independent enforcement layers on the B2B path:

1. **Request rate limiting** uses a 60-second sliding window. In development/test it uses a bounded in-memory adapter; production requires Redis/Valkey-compatible `REDIS_URL` so limits are shared across workers.
2. **Daily organization quota** uses the plan record and usage ledger. The initial migration seeds Free at 100 accepted scan requests/day and Pro at 5,000/day. The `/v1/scans` route reserves a usage row in the same database transaction as the quota check.

A minute-level rate limit is not a daily entitlement, and neither is a substitute for the other.

## Endpoints

- `POST /v1/scans`: requires `scans:write` scope, validates the upload, reserves one unit against the organization's daily plan limit, then streams the analysis response.
- `GET /v1/usage`: requires `usage:read` scope and returns the authenticated organization's plan, UTC date, used units, limit, remaining units, and reset time.
- API keys use `X-API-Key`; JWTs use `Authorization: Bearer`.

The existing unversioned `/upload-image-stream` and `/upload-image` routes remain public for the current demo and do not use organization metering. They are still subject to the middleware's per-IP rate limit. Do not expose the legacy routes as the paid B2B contract; deprecation or a separate public-demo policy is still required.

## Configuration

- `RATE_LIMIT_REQUESTS_PER_MINUTE`: integer from 1 to 100000; default 60.
- `REDIS_URL`: optional in development/test; required in production and must use `redis://` or `rediss://`.
- `DATABASE_URL`: PostgreSQL in production; SQLite is supported for local/test use.
- Production fails closed with HTTP 503 if shared rate-limit storage is unavailable. Non-production falls back only when Redis is not configured; an explicitly configured but unavailable Redis does not silently become an in-memory limiter.

## Implementation details

- Redis uses an atomic Lua sliding-window operation and expires sorted-set keys.
- The in-memory adapter is bounded and intended only for single-process development/test, not multi-worker production.
- The middleware is pure ASGI so it does not buffer SSE response bodies.
- Rate limiting uses the peer IP plus a hashed credential identity when present. It does not trust arbitrary `X-Forwarded-For` values. Deployments behind a reverse proxy must configure a trusted proxy/client-IP policy at the ASGI edge; otherwise requests may share the proxy's IP bucket.
- Daily quota reservations lock the organization row in PostgreSQL before checking usage and inserting the pending record. This serializes quota decisions for one organization across workers.
- A pending usage reservation counts while recent; abandoned pending reservations older than ten minutes are ignored. Server failures (HTTP 5xx) are not charged by the current usage query; accepted success/client-aborted requests count.
- The daily reset is UTC midnight, not the customer's local timezone.
- Request IDs, API-key IDs, endpoints, status, and units are stored. Image bytes and nutrition payloads are not stored in usage records.

## Known limitations and release gates

- The in-memory rate limiter is not shared across workers and is not a production substitute for Redis.
- No separate IP reputation, bot detection, WAF, or DDoS provider is configured by application code.
- Quotas currently apply to authenticated `/v1/scans`; legacy demo routes are not charged against organization usage.
- The versioned scan route is the B2B path; connect the public demo or portal to it only after a deliberate authentication/product decision.
- Load-test concurrent requests near a quota boundary against PostgreSQL and Redis before production.
- Configure and test trusted proxy behavior, Redis availability alerts, and database pool limits in the target deployment.
- Add a support/admin override process with audit history before changing plan entitlements manually.
