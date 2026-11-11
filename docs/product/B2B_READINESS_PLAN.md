# ActualPlate B2B Readiness Plan

This plan is the implementation backlog for ActualPlate's public website, image-analysis API, nutrition-provider integration, and future B2B SaaS controls. Status is evidence-based; planned work is not described as released.

## Delivery rules

- All work stays on topic branches and is submitted as reviewable pull requests. Never merge to `main` without the repository owner's explicit instruction.
- A task is complete only when its acceptance criteria and relevant automated checks pass.
- Keep the current public website honest about unavailable functionality. Do not publish fake uptime, accuracy, customer, pricing, security certification, or SLA claims.
- Do not fabricate measurements. Missing or scientifically unsupported mass/volume/nutrition values must remain unavailable.
- Keep API credentials server-side. Do not expose provider secrets, raw provider error bodies, user images, or internal tracebacks in public responses.
- Separate application code from controls requiring provider accounts or operational access, such as DNS, WAF, GPU deployment, billing accounts, email, monitoring, and legal approvals.

## P0 — Product truth and scientific validity

- [ ] Replace hardcoded food aliases with deterministic matching where appropriate; test ambiguous and low-confidence matches.
- [ ] Fix bounding-box axis/dimension bugs and add non-square image regression fixtures.
- [ ] Assess relative-thickness geometry against measured ground truth; do not treat monocular depth as metric scale by default.
- [ ] Evaluate porosity/occlusion gates for leafy or highly porous foods; return unknown when evidence is insufficient.
- [ ] Evaluate high-volume plausibility gates against labeled data; thresholds must be documented, tested, and configurable when justified.
- [ ] Pin every deployable model artifact/revision and record model, code, and configuration versions.
- [ ] Build a labeled, independently measured validation set; report error distributions, subgroup performance, confidence intervals, and failure cases.
- [ ] Add drift monitoring only after stable evaluation metrics and a representative reference set exist.
- [ ] Keep authoritative `volume_cm3` null until an independently validated metric-scale method exists.
- [ ] Keep mass and nutrition unavailable when volume, density, or serving scale is not trustworthy.

## P0 — Data, identity, and access control

- [ ] Select production database and transaction model; use SQLAlchemy and versioned Alembic migrations.
- [ ] Create User, Organization, Membership, APIKey, UsageLog, AuditLog, and Plan/Entitlement records.
- [ ] Implement signup/login with secure password hashing, account verification, and recovery policy.
- [ ] Implement short-lived JWT access tokens with validated issuer, audience, expiry, and signing-key rotation policy.
- [ ] Generate cryptographically random, prefixed API keys; show secrets once and persist only keyed hashes.
- [ ] Implement API-key scopes, rotation, revocation, last-used metadata, and expiry policy.
- [ ] Resolve the authenticated principal through dependencies; remove hardcoded/shared-token assumptions from customer-facing paths.
- [ ] Implement role-based authorization and organization-level tenant isolation on every read/write.
- [ ] Add negative tests for cross-tenant access, revoked keys, expired tokens, missing scopes, and role escalation.
- [ ] Add CSRF/session protection if browser-cookie authentication is selected; keep browser session tokens out of local storage where practical.

## P0 — Abuse prevention and application security

- [ ] Implement shared Redis-backed rate limits for production and a deterministic in-memory test/local adapter.
- [ ] Enforce plan entitlements with atomic database-backed quota checks; never rely on frontend limits.
- [ ] Validate image magic bytes and decoder success; reject unsupported formats, oversized files, oversized decoded dimensions, and malformed images.
- [ ] Re-encode accepted images to canonical pixel data and discard unnecessary metadata before inference.
- [ ] Treat text visible in an image as untrusted content; prevent model prompts from following image-embedded instructions.
- [ ] Hide interactive API docs in production unless deliberately exposed behind an access policy.
- [ ] Restrict CORS to configured, exact origins and validate the deployed frontend origin.
- [ ] Add HTTP security headers and deployment-level HSTS once HTTPS is confirmed.
- [ ] Add privacy-aware audit events for authentication, key lifecycle, administrative actions, and billing changes.
- [ ] Define retention/deletion policy for images, logs, derived data, and provider responses.
- [ ] Configure WAF/DDoS protection at the selected edge provider and test operational runbooks.

## P0 — FatSecret integration and nutrition data handling

- [ ] Keep FatSecret credentials in server-side secret storage and load configuration without import-time crashes in unrelated test/dev paths.
- [ ] Validate OAuth token responses, expiry values, timeout configuration, and token refresh behavior.
- [ ] Handle provider timeouts, authentication failures, rate limits, malformed responses, and upstream outages with safe structured errors.
- [ ] Never return raw provider response bodies, credentials, or upstream diagnostics to public clients.
- [ ] Validate food IDs and search expressions; cap query lengths and response sizes.
- [ ] Normalize single-object and list serving responses and reject invalid/non-finite nutrition numbers.
- [ ] Scale nutrition only against a valid gram-based reference serving and a valid estimated food mass.
- [ ] Add bounded caching only for content the current FatSecret terms explicitly permit to be cached; respect refresh/deletion and attribution requirements.
- [ ] Add request coalescing/backoff where useful without bypassing FatSecret rate limits.
- [ ] Show required FatSecret attribution wherever FatSecret content is displayed and on the public website.
- [ ] Link and comply with current FatSecret API terms and attribution policy; verify permissions and market access before commercial launch.
- [ ] Add contract tests with mocked responses; perform credentialed integration tests only in a protected environment with authorized keys.
- [ ] Do not describe ActualPlate as endorsed by FatSecret or promise access to data/features beyond the current provider agreement.

## P1 — API contract and developer experience

- [ ] Standardize public routes under `/v1/` and define stable request/response schemas.
- [ ] Define consistent error envelopes, request IDs, pagination, timeouts, and cancellation behavior.
- [ ] Implement `GET /v1/system/health` and readiness semantics that distinguish API, database, provider, and GPU/model status.
- [ ] Implement protected warmup behavior only for deployments where GPU warmup is supported and authorized.
- [ ] Define sandbox credentials, isolated test data, separate quotas, and no real billing/provider side effects.
- [ ] Generate and test a Postman collection against the published API contract.
- [ ] Publish a Python SDK first, then Node.js SDK after API schemas stabilize.
- [ ] Add signed webhooks only when there is a durable asynchronous job lifecycle, with retries, idempotency, replay protection, and delivery history.
- [ ] Add request/response observability with redaction of credentials, image data, and sensitive nutrition records.

## P1 — Website and B2B conversion

- [ ] Publish a responsive B2B landing page that explains the product, supported workflow, integration surface, limitations, and roadmap truthfully.
- [ ] Separate the public marketing site from the interactive scanner/demo experience.
- [ ] Configure API base URL at build time and validate it; remove dead configuration placeholders and hardcoded frontend endpoints.
- [ ] Add a dependency-free web build/test pipeline to CI.
- [ ] Add accessible navigation, mobile layouts, keyboard focus states, reduced-motion support, meaningful error states, and loading/cancellation behavior.
- [ ] Show missing numerical values as unavailable rather than converting them to zero.
- [ ] Add a developer documentation page with authentication, request examples, response schemas, errors, rate limits, and versioning once those contracts exist.
- [ ] Add honest integration/contact workflow without inventing email addresses or claiming an onboarding process that is not operational.
- [ ] Add privacy notice, terms, data retention, and image-handling disclosures before accepting production customer data.
- [ ] Add customer dashboard only after identity, organizations, API keys, and metering APIs are stable.
- [ ] Add usage graphs, key management, team management, and billing views only when backed by real APIs.

## P1 — Billing, plans, and organizations

- [ ] Define plan/entitlement data model before setting public prices or plan promises.
- [ ] Implement `GET /v1/usage` using persisted, tenant-scoped usage records.
- [ ] Integrate Stripe or Razorpay only after the business chooses a provider and merchant account.
- [ ] Verify payment webhook signatures and enforce idempotent subscription state transitions.
- [ ] Implement overage billing only after quota and metering correctness are load-tested.
- [ ] Generate invoices and tax/GST calculations after accounting and jurisdiction requirements are confirmed.
- [ ] Implement organization invites, membership roles, and seat/usage ownership.
- [ ] Add coupon/promo logic only after subscription state and audit records exist.

## P1/P2 — Architecture, operations, and deployment

- [ ] Refactor into domain modules incrementally where it improves ownership and testing; avoid a folder-only rewrite.
- [ ] Add structured request logs, correlation IDs, redaction, metrics, and actionable alerts.
- [ ] Add managed secret storage for production and documented local development configuration.
- [ ] Configure domain, DNS, HTTPS, reverse-proxy timeouts, request limits, and deployment health checks.
- [ ] Deploy backend and GPU inference service to a selected provider after runtime/model compatibility tests.
- [ ] Configure frontend hosting and environment-specific API base URLs.
- [ ] Add uptime monitoring and error tracking after production ownership and alert routing are defined.
- [ ] Add k6/Locust load tests for API, quotas, provider rate limits, and concurrent inference.
- [ ] Evaluate multi-region deployment only after measured latency, data residency, and operational requirements justify it.
- [ ] Define model fallback behavior based on tested quality and explicit failure semantics.
- [ ] Add Celery/Redis or another task queue only when durable asynchronous workloads require it.
- [ ] Define image/log cold storage, retention, access control, deletion, and cost policy.

## P2 — Portal and advanced capabilities

- [ ] Build customer portal for onboarding, API keys, usage, support, and organization membership.
- [ ] Build internal admin operations with least-privilege RBAC, audit trails, and break-glass policy.
- [ ] Add email delivery after choosing a provider and verifying sender domain, consent, and deliverability.
- [ ] Add pricing tiers only after capacity, support costs, provider licensing, and quota economics are validated.
- [ ] Add public status page and incident communication process.
- [ ] Evaluate active-learning and human-labeling workflow against data rights, consent, and retention policy.
- [ ] Evaluate custom VLM fine-tuning only if measured evaluation demonstrates a concrete gap.
- [ ] Evaluate ARCore/ARKit and native mobile clients as optional capture improvements, not substitutes for current scientific validation.
- [ ] Add prompt/model A/B testing only with controlled experiments, privacy review, and versioned metrics.
- [ ] Establish SOC 2 readiness evidence and control ownership before claiming compliance.

## P2 — Legal, privacy, and commercial readiness

- [ ] Draft Terms of Service and Privacy Policy with qualified legal review.
- [ ] Assess India DPDP Act and GDPR applicability, lawful basis, user rights, cross-border processing, and retention.
- [ ] Review FatSecret license, attribution, storage, branding, and commercial-use obligations with current terms.
- [ ] Evaluate trademark availability and registration for ActualPlate.
- [ ] Obtain legal review before any patent/NDAs/IP ownership claims.
- [ ] Set up business accounting, invoicing, tax, and bank operations with qualified professionals.
- [ ] Prepare truthful B2B outreach, technical sales material, and engineering documentation.
- [ ] Select CRM only when a real sales process and data-handling policy exist.

## Execution order

1. Stabilize dependency, configuration, upload, API error, and pipeline contracts (active engineering PRs).
2. Make the website truthful, configurable, accessible, and build-tested; audit FatSecret handling and attribution.
3. Finish scientific validity boundaries and independent ground-truth evaluation.
4. Introduce database/migrations, organization identity, API keys, authentication, authorization, audit logs, and tenant isolation.
5. Add atomic usage metering, entitlements, and shared rate limiting.
6. Publish versioned API contracts, SDK/Postman docs, health/readiness, and sandbox semantics.
7. Add operational controls, monitoring, deployment, and load testing.
8. Add billing, portal, webhooks, email, and admin workflows after their backend dependencies.
9. Complete legal/compliance readiness with qualified professionals before commercial launch.

## Current web work in this branch

- [x] Replace the single-purpose scan screen with a B2B-oriented landing page plus a clearly labeled interactive demo.
- [x] Describe current and planned platform capabilities without presenting the roadmap as released.
- [x] Make the frontend API base URL a build-time configuration value.
- [x] Avoid rendering missing numeric outputs as zero.
- [x] Add FatSecret attribution and links to the current official terms and attribution policy.
- [x] Add dependency-free frontend build contract tests and a dedicated Web CI workflow.
- [x] Confirm the Web CI run passes on PR #62 head `0c59f3e` (Web CI and Python CI passed).
- [ ] Review current FatSecret client error handling, OAuth token lifecycle, normalization, and integration tests in a separate backend PR.
- [ ] Verify the live hosted website and deployed API configuration after repository changes are reviewed and deployed.
