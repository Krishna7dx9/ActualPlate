# B2B identity and API-key security

## Scope of the first identity foundation

This change adds SQLAlchemy models and an Alembic migration for users, plans, organizations, memberships, API keys, usage records, and audit records. It adds initial signup/login and API-key management endpoints.

Routes introduced:
- `POST /v1/auth/signup`
- `POST /v1/auth/login`
- `GET /v1/auth/me`
- `POST /v1/api-keys`
- `GET /v1/api-keys`
- `DELETE /v1/api-keys/{key_id}`

Signup is disabled in production until email verification and signup abuse controls are implemented. Production login rejects accounts that have not been verified. Do not advertise this foundation as a complete production identity service yet.

## Configuration

Required for production:
- `DATABASE_URL`: PostgreSQL URL; `postgresql://...` is normalized to the installed psycopg 3 driver.
- `JWT_SECRET_KEY`: at least 32 UTF-8 bytes, managed as a secret and rotated under a documented procedure.
- `API_KEY_HASH_SECRET`: separate secret of at least 32 UTF-8 bytes, used to HMAC API keys before persistence.
- `APP_ENV=production`.
- Existing production vision token, upload limits, and exact CORS origins.

Optional:
- `ACCESS_TOKEN_TTL_SECONDS`: 60–3600 seconds; defaults to 900 seconds.
- `DATABASE_URL=sqlite:///./actualplate.db` for local development. Tests should use an isolated in-memory SQLite engine.

Never use the test secrets in a deployment. Never commit real secrets to the repository or put them in browser code.

## Password and token policy

- Passwords are hashed with bcrypt; accepted password length is 12–72 UTF-8 bytes to avoid silent bcrypt truncation.
- JWTs use HS256 with a configured signing key, issuer, audience, expiry, not-before, issue time, unique token ID, and explicit access-token type.
- JWTs are short-lived. Refresh tokens, password reset, email verification, account recovery, MFA, and key rotation for JWT signing are not included in this initial slice.
- Login failures use a generic invalid-credentials message.
- JWT role and organization access are re-checked against active database membership on each request; the token's role is not trusted as authorization evidence.

## API-key policy

- Keys are cryptographically random, prefixed by environment, shown only in the creation response, and stored only as an HMAC-SHA256 digest.
- The HMAC key is separate from the JWT signing key.
- API keys are organization-scoped, have explicit scopes, a default 90-day expiry, revocation, and last-used tracking.
- Supported scopes in this initial version are `scans:write`, `usage:read`, and `keys:manage`.
- API keys are supplied in the `X-API-Key` header. JWT access tokens use `Authorization: Bearer`.
- Revoked, expired, inactive, or orphaned keys are rejected. The creator must retain an active membership in the key's organization.
- Key listing never returns the raw key or its digest.

## Authorization and tenant isolation

- Roles are `owner`, `admin`, and `member`.
- Signup creates an organization and owner membership.
- API-key management is restricted to owners/admins and also requires the `keys:manage` scope for API-key-authenticated calls.
- All key list/revoke queries are constrained by the authenticated organization ID; a foreign organization's key is reported as not found.
- This does not yet protect the existing legacy upload/nutrition endpoints with B2B API-key auth. Protecting customer scan routes, enforcing quotas, and attaching usage records are follow-up work.

## Migrations and plan limits

Run schema changes with Alembic; do not call `Base.metadata.create_all` in production. The initial migration seeds:
- `free`: 100 requests/day
- `pro`: 5,000 requests/day

The plan limits are stored as database records, but request metering and atomic quota enforcement are not yet implemented. Do not treat those values as enforced limits until the metering/rate-limit workstream is complete.

## Release blockers still open

- [ ] Email verification and recovery flow.
- [ ] Signup and login rate limiting, anti-automation controls, and abuse monitoring.
- [ ] Production PostgreSQL deployment, backups, migration rehearsal, and restore test.
- [ ] API-key auth applied to versioned scan endpoints.
- [ ] Atomic usage metering and plan-limit enforcement.
- [ ] Audit-log retention/access policy and operational alerting.
- [ ] Security review for token/key rotation, account takeover, and organization lifecycle.
- [ ] Protected live deployment smoke tests and independent review.

These endpoints are a foundation for review and tests, not a claim of complete enterprise readiness.
