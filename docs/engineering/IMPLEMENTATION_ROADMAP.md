# Engineering Implementation Roadmap

This roadmap is the execution record for the engineering improvement program. It is dependency-ordered. Work is split into focused pull requests so reviewers can inspect and validate each change independently.

## Working agreements

- All implementation happens on topic branches.
- Pull requests are reviewed by the repository owner; none are merged by this automation.
- No credentials, secrets, or environment-specific endpoints are committed.
- Runtime configuration is validated at startup or request boundaries. Algorithmic constants remain only where they are part of a documented method and have tests or scientific justification.
- A passing unit test is not evidence of real-world metric accuracy.
- Each PR states exactly what was inspected, changed, and executed. Unavailable checks are marked pending.

## Workstream status

| ID | Workstream | Priority | Acceptance criteria | Status |
|---|---|---|---|---|
| ENG-001 | Repository baseline audit | P0 | Current structure, verified findings, constraints, and remediation order documented | PR #53 open |
| ENG-002 | Docker dependency manifest integrity | P0 | Dockerfile references tracked dependency manifests; Docker image build verified in CI | PR #61 open; Docker build passed on pre-sync head; verification rerunning after stacked-branch synchronization |
| ENG-003 | CI dependency reproducibility | P0 | CI installs from tracked manifests and validates dependency consistency without fake production credentials | PR #61 open; Python CI passed on pre-sync head; verification rerunning after stacked-branch synchronization |
| ENG-004 | Backend configuration contract | P0 | Typed settings; required/optional variables explicit; invalid configuration fails clearly | PR #56 open; Python CI passed after dependency pin (2026-11-12) |
| ENG-005 | Vision authentication contract | P0 | Token policy consistent between docs, startup, and request handling; no import-time surprise | PR #55 open; Python CI passed after dependency pin (2026-11-12); extended by PR #60 |
| ENG-006 | API error contract and logging | P1 | Stable public errors; details logged safely; no raw exception leakage | PR #56 open; Python CI passing |
| ENG-007 | CORS policy | P1 | Explicit environment configuration; restrictive default; no wildcard policy in production | PR #56 open; Python CI passing |
| ENG-008 | Upload request limits and decoding | P0 | Bounded request/file handling; actual image decoding; supported-format policy and resource-limit tests | PR #56 open; Python CI passing |
| ENG-009 | Backend execution/concurrency | P1 | Blocking work isolated with bounded capacity; cancellation and shutdown behavior tested | PR #57 open; Python CI passing |
| ENG-010 | Provider failure semantics | P1 | Nutrition-provider failures visible as structured partial failures; no silent broad exception swallowing | PR #57 open; Python CI passing |
| ENG-011 | Service API contract | P1 | Typed request/response models and contract tests shared across service boundary | Not started |
| ENG-012 | Vision volume semantics | P0 | Response distinguishes relative geometry, metric scale evidence, estimate, uncertainty, and validity | PR #60 open; Python CI passing |
| ENG-013 | Single authoritative volume path | P0 | One production volume path; deprecated alternatives removed or explicitly diagnostic | PR #60 open; Python CI passing |
| ENG-014 | Metric-scale feasibility study | P0 | Assumptions, identifiability limits, and candidate methods documented and experimentally assessed | Not started |
| ENG-015 | Independent volume evaluation | P0 | Analytic fixtures plus measured ground-truth protocol; metrics and confidence intervals reported | Not started |
| ENG-016 | Vision model lifecycle | P1 | Lazy loading, readiness/liveness, device selection, and resource ownership tested | Partial; GPU model-load verification not run |
| ENG-017 | Data persistence assessment | P2 | Storage requirements and data lifecycle documented before introducing a database | Not started |
| ENG-018 | Production observability | P2 | Structured logs, request correlation, health semantics, and actionable metrics | Partial; request IDs and structured failure codes exist, broader metrics pending |
| ENG-019 | Deployment hardening | P2 | Reproducible image, non-root runtime where feasible, health checks, resource policy, and deployment guide | PR #61 adds dependency separation; Docker build passed on pre-sync head and is rerunning after base synchronization; runtime hardening remains |
| ENG-020 | Final engineering verification | P0 | Full supported test suite, image build, security checks, and known limitations recorded | Not started |
| ENG-021 | Persistence architecture and migrations | P0 | Select PostgreSQL production store; SQLAlchemy models, Alembic migrations, transaction boundaries, and test database strategy documented | Not started |
| ENG-022 | Identity and account lifecycle | P0 | User/org/membership schema; signup/login; secure password hashing; email verification and account recovery policy | Not started |
| ENG-023 | API key lifecycle | P0 | Cryptographically random prefixed keys; hash-only persistence; one-time display; scoped permissions, rotation, revocation, and last-used tracking | Not started |
| ENG-024 | Authentication and authorization | P0 | Dependency-based principal resolution; JWT access-token policy; organization isolation; role and permission checks on every protected route | Not started |
| ENG-025 | Usage metering and entitlements | P0 | Transactional request accounting; plan entitlements; atomic quota enforcement; usage API; idempotency and auditability | Not started |
| ENG-026 | Rate limiting and abuse controls | P0 | Shared Redis-backed limits in production; safe local/test backend; per-key/user/org limits; tier policy; abuse response and observability | Not started |
| ENG-027 | Application security hardening | P0 | Security headers; production docs policy; metadata-stripping image re-encode; prompt/data trust boundaries; audit events; threat model | Partial; CORS and bounded upload validation are covered by PR #56 |
| ENG-028 | Versioned public API contract | P1 | Versioned routes, typed schemas, stable error envelope, pagination, deprecation policy, contract tests | Not started |
| ENG-029 | Developer integration experience | P1 | OpenAPI examples, Postman collection, sandbox semantics, readiness/health endpoints, SDK release process | Not started |
| ENG-030 | Async jobs and webhook delivery | P2 | Queue selected only when workload requires it; durable job state, retries, signatures, replay protection, delivery logs | Not started |
| ENG-031 | Billing and organization plans | P2 | Provider integration, signed payment webhooks, subscription state machine, tax/invoice requirements, reconciliation | Not started |
| ENG-032 | Operations and release controls | P2 | Secret manager, Sentry/metrics, uptime, load tests, deployment health checks, backup/retention, regional strategy | Partial; Docker build CI passed, target GPU runtime validation remains |
| ENG-033 | Portal and internal operations | P2 | Secure customer portal, usage/billing views, key management, admin RBAC, support/audit workflow | Not started |
| ENG-034 | Scientific model lifecycle | P0 | Versioned model artifacts, reproducible evaluation set, drift monitoring tied to ground-truth metrics, fallback policy | Not started |
| ENG-035 | Legal and compliance readiness | P2 | Privacy/terms/data-processing review, India DPDP/GDPR applicability assessment, retention and deletion process, SOC 2 readiness evidence | Not started; requires qualified legal/compliance review

## Execution sequence

1. **Foundation:** ENG-001 through ENG-005. Establish reproducible builds, CI, and explicit service configuration.
2. **Backend safety:** ENG-006 through ENG-011. Stabilize API behavior, upload handling, concurrency, provider semantics, and service contracts.
3. **Scientific correctness:** ENG-012 through ENG-016 and ENG-034. Resolve the volume contract, metric-scale feasibility, independent evaluation, model pinning, and drift evidence before claiming physical validity.
4. **Identity and platform security:** ENG-021 through ENG-027. Build persistence, identity, API keys, authorization, metering, shared rate limits, and security controls as one coherent foundation.
5. **Developer and commercial platform:** ENG-028 through ENG-033. Add versioned APIs, SDKs, sandbox semantics, jobs/webhooks, billing, portal, and operational controls only after their dependencies are stable.
6. **Release and compliance review:** ENG-020 and ENG-035. Run the supported verification suite and complete legal/compliance review with qualified professionals. Website and provider integration work is tracked in ENG-036 through ENG-038.

## Scope discipline

The earlier task inventory is a broad checklist, not permission to add unnecessary infrastructure. For example, a database, cache, queue, or microservice should be introduced only when the product's data and workload requirements justify it. A task may be closed as **not applicable** with a recorded reason after inspection.

## Verification record

GitHub Actions provides evidence for the specific commits and jobs linked from each PR. A passing Python CI run does not prove Docker runtime startup, CUDA model loading, live provider integration, or physical volume accuracy. Those checks must be reported separately. The Docker image build workflow passed on PR #61 commit 53f17e2. This verifies image construction only; it does not verify CUDA model loading, GPU inference, deployment startup under production secrets, or physical volume accuracy. Target-hardware and real-world measurement validation remain separate release gates.


## Master product backlog — implementation guardrails

The requested product checklist is tracked through the workstream IDs above. These rules prevent premature infrastructure and unsafe claims:

- **Authentication model:** B2B API-key authentication is the first machine-to-machine path. User login/JWT supports the dashboard and key management; it does not replace API-key scopes. Never store raw API keys or passwords.
- **Authorization model:** organization-scoped resources are mandatory. Every data lookup and mutation must enforce tenant ownership before object access; role checks alone are not sufficient.
- **Quota model:** plan limits are entitlements, not scattered constants. Usage increments and quota decisions must be atomic so concurrent requests cannot bypass limits. Redis rate limiting and database usage metering solve different problems.
- **Image safety:** verify magic bytes and decode; then re-encode to a canonical format before model processing. Preserve only necessary color/orientation information and explicitly discard metadata. File size and pixel limits remain separate controls.
- **Prompt safety:** treat all text visible in an uploaded image as untrusted user content, never as system/developer instructions. OCR stripping alone is not a complete prompt-injection defense.
- **Sandbox:** sandbox requests must use isolated credentials, data, quotas, and provider behavior. A request header must not silently turn production credentials or real billing into a test environment.
- **Billing:** usage accounting must be correct before payment integration. Payment webhooks require signature verification, idempotency, replay protection, and reconciliation; do not grant plans from an unverified callback.
- **Infrastructure:** WAF, DDoS protection, TLS/domain, GPU deployment, monitoring, and secret management are deployment controls and require selected providers/accounts. Do not claim they are implemented by application code alone.
- **Architecture:** introduce DDD boundaries incrementally where domain ownership is real. Do not perform a folder-only rewrite. Use a queue only after latency, reliability, or workload requirements justify asynchronous jobs.
- **ML validity:** difflib matching is a deterministic text-matching utility, not a substitute for scientific evaluation. Thickness scaling, porosity/plausibility gates, and drift alerts must be justified against labeled ground truth and must return unknown/null rather than fabricated measurements.
- **Commercial/legal work:** pricing, GST/tax, ToS, Privacy Policy, DPDP/GDPR applicability, NDAs, patents, trademarks, and SOC 2 require business decisions and qualified legal/accounting/compliance review. Engineering can prepare evidence and technical controls but cannot certify legal compliance.
