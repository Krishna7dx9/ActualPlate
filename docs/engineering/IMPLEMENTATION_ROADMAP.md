# Engineering Implementation Roadmap

This roadmap is the execution record for the engineering improvement program. It is intentionally dependency-ordered. Work is split into focused pull requests so reviewers can inspect and validate each change independently.

## Working agreements

- All implementation happens on topic branches.
- Pull requests target `main` for review; none are merged by the automation.
- No credentials, secrets, or environment-specific endpoints are committed.
- Runtime configuration is validated at startup or request boundaries. Algorithmic constants remain only where they are part of a documented method and have tests or scientific justification.
- A passing unit test is not evidence of real-world metric accuracy.
- Each PR states exactly what was inspected, changed, and executed. Unavailable checks are marked pending.

## Workstream status

| ID | Workstream | Priority | Acceptance criteria | Status |
|---|---|---|---|---|
| ENG-001 | Repository baseline audit | P0 | Current structure, verified findings, constraints, and remediation order documented | PR #53 open |
| ENG-002 | Docker dependency manifest integrity | P0 | Dockerfile references only tracked dependency manifests; build verified in CI | PR #54 open; build pending |
| ENG-003 | CI dependency reproducibility | P0 | CI installs from tracked manifests and uses explicit test extras without fake production credentials | Not started |
| ENG-004 | Backend configuration contract | P0 | Typed settings; required/optional variables explicit; invalid configuration fails clearly | Not started |
| ENG-005 | Vision authentication contract | P0 | Token policy consistent between docs, startup, and request handling; no import-time surprise | Not started |
| ENG-006 | API error contract and logging | P1 | Stable public errors; details logged safely; no raw exception leakage | Not started |
| ENG-007 | CORS policy | P1 | Explicit environment configuration; restrictive default; no wildcard policy in production | Not started |
| ENG-008 | Upload request limits and decoding | P0 | Bounded request/file handling; actual image decoding; cleanup and error tests | Not started |
| ENG-009 | Backend execution/concurrency | P1 | Blocking work isolated with bounded capacity; cancellation and shutdown behavior tested | Not started |
| ENG-010 | Provider failure semantics | P1 | Nutrition-provider failures visible as structured partial failures; no silent broad exception swallowing | Not started |
| ENG-011 | Service API contract | P1 | Typed request/response models and contract tests shared across service boundary | Not started |
| ENG-012 | Vision volume semantics | P0 | Response distinguishes relative geometry, metric scale evidence, estimate, uncertainty, and validity | Not started |
| ENG-013 | Single authoritative volume path | P0 | One production volume path; deprecated alternatives removed or explicitly diagnostic | Not started |
| ENG-014 | Metric-scale feasibility study | P0 | Assumptions, identifiability limits, and candidate methods documented and experimentally assessed | Not started |
| ENG-015 | Independent volume evaluation | P0 | Analytic fixtures plus measured ground-truth protocol; metrics and confidence intervals reported | Not started |
| ENG-016 | Vision model lifecycle | P1 | Lazy loading, readiness/liveness, device selection, and resource ownership tested | Not started |
| ENG-017 | Data persistence assessment | P2 | Storage requirements and data lifecycle documented before introducing a database | Not started |
| ENG-018 | Production observability | P2 | Structured logs, request correlation, health semantics, and actionable metrics | Not started |
| ENG-019 | Deployment hardening | P2 | Reproducible image, non-root runtime where feasible, health checks, resource policy, and deployment guide | Not started |
| ENG-020 | Final engineering verification | P0 | Full supported test suite, build, security checks, and known limitations recorded | Not started |

## Execution sequence

1. **Foundation:** ENG-001 through ENG-005. Establish reproducible builds, CI, and explicit configuration/authentication contracts.
2. **Backend safety:** ENG-006 through ENG-011. Stabilize API behavior, upload handling, concurrency, provider semantics, and service contracts.
3. **Scientific correctness:** ENG-012 through ENG-016. Resolve the volume contract and scale assumptions before claiming physical validity.
4. **Operations:** ENG-017 through ENG-019. Make data retention and production operations explicit based on actual product needs.
5. **Release review:** ENG-020. Run the full available verification suite and publish remaining risks.

## Scope discipline

The earlier task inventory is a broad checklist, not permission to add unnecessary infrastructure. For example, a database, cache, queue, or microservice should be introduced only when the product's data and workload requirements justify it. A task may be closed as **not applicable** with a recorded reason after inspection.

## Verification record

GitHub source inspection is not equivalent to running the code. At the time this roadmap was created, the repository has not been cloned into an execution environment in this work session; local test runs, Docker builds, GPU inference, and physical measurement validation remain unverified.
