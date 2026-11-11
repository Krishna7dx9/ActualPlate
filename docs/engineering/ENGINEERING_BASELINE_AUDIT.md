# Engineering Baseline Audit

**Audit target:** `main` at `8788991f605f1ac100840701bfc355c74c89e2b2`  
**Audit branch:** `docs/engineering-baseline-audit`  
**Scope:** Repository metadata, tracked file tree, architecture documentation, CI, deployment files, backend request path, vision inference path, and representative tests.

This document records observations from repository inspection. It is not a claim that the application has been executed or that production behavior has been independently verified.

## 1. Repository state

- The default branch is `main`.
- GitHub reports `main` as protected.
- An existing branch, `feat/dockerization`, exists at `98a154e5992191ddabf07496e861bc52fc50e9c9`. It was not modified.
- No open pull requests were returned during this audit.
- The repository already contains architecture notes, two ADRs, Docker files, a CI workflow, benchmarks, and multiple test categories. These were inspected rather than recreated.

## 2. Confirmed findings

### Critical: deployment build references a missing file

- `Dockerfile` contains `COPY backend/requirements.txt /app/backend/requirements.txt`.
- The tracked repository tree does not contain `backend/requirements.txt`.
- A Docker build using this Dockerfile is therefore expected to fail at that COPY instruction unless the file is supplied by an untracked build context. This must be resolved before treating the image as buildable.
- The Dockerfile separately copies `vision_service/inference/requirements.txt`, which does exist.

### High: scale and volume status are not evidence of metric correctness

- `vision_service/inference/api.py` calculates volume from a normalized depth map and VLM-supplied thickness and diameter values.
- The scale values originate in `vision_service/recognition/food_recognition.py`, whose prompt asks a vision-language model to estimate physical dimensions from the image.
- A result is marked `volume_status = "valid"` when these estimated values are present and the calculation completes. That status currently describes pipeline completion, not independently validated metric accuracy.
- `vision_service/inference/inference.py` describes MoGe output as metric depth and labels the returned unit as meters. The architecture ADR correctly notes that MoGe-2 does not supply absolute metric scale out of the box. These statements need a single, technically accurate contract.
- `vision_service/inference/volume.py` estimates volume by fitting a supporting plane from background pixels. It requires enough visible supporting-surface pixels and a sufficiently planar fit. This method is not compatible with every intended capture condition, especially images with no visible planar support.
- The API imports `estimate_volume_cm3` and exposes debug documentation for it, but the inspected `/detect` execution path computes volume using a separate normalized-depth and VLM-scale calculation. The two methods currently represent competing approaches.

### High: vision service authentication contract is contradictory

- `vision_service/inference/api.py` raises during module import when `VISION_API_TOKEN` is absent.
- The same module's `_check_auth` function contains an optional-token branch for when the token is absent. That branch cannot be reached in the current module initialization path.
- The documented development/production behavior and the actual import behavior must be reconciled.

### High: request processing performs blocking work in async handlers

- `backend/app/services/pipeline.py` exposes async request functions but calls synchronous model/provider functions and synchronous HTTP operations.
- `backend/app/api/routes/upload.py` invokes these functions from async FastAPI routes.
- This can block the event loop and reduce concurrent request capacity. The right fix depends on the model runtime and provider client lifecycle; it should not be replaced with arbitrary thread usage without measuring and defining concurrency limits.

### High: upload validation trusts client-supplied MIME type

- Both backend and vision upload routes check the declared content type.
- The inspected code does not establish a request-size limit or verify the actual decoded image format before writing/processing the upload.
- The backend pipeline reads the full upload into memory. The vision service copies the upload to a temporary file.
- Validation should be based on bounded streaming reads and actual image decoding, with explicit supported formats and resource limits.

### Medium: model and geometry behavior contains embedded policy values

- `vision_service/inference/api.py` contains an in-code alias mapping for labels and substring matching rules.
- `vision_service/inference/inference.py` uses default detection thresholds and a fallback focal length value when predicted intrinsics are unavailable and no override is supplied.
- `vision_service/inference/volume.py` embeds sampling limits, plane-fit thresholds, dilation sizes, iteration counts, and a deterministic random seed.
- Some numerical constants are algorithm parameters and may legitimately remain in the algorithm with documented rationale. Operational policies and tunable values should be centralized, typed, validated, and surfaced through configuration. No parameter should be changed merely to make a benchmark pass.

### Medium: nutrition failures can disappear silently

- `backend/app/services/pipeline.py` catches broad exceptions around nutrition lookup and discards them.
- A successful geometry result can therefore be returned without a visible indication that nutrition retrieval failed.
- The response contract should distinguish geometry success from nutrition-provider failure and expose safe, structured diagnostics without leaking secrets or upstream internals.

### Medium: global exception response includes raw exception text

- `backend/app/main.py` returns `str(exc)` in the JSON error response.
- Raw exception messages may disclose implementation details or sensitive upstream information. The API should return a stable public error envelope and retain detailed context only in protected logs.

### Medium: CORS policy is unrestricted

- `backend/app/main.py` allows all origins, methods, and headers.
- Credentials are disabled, but a production origin policy should be explicitly configured per environment rather than left globally permissive.

### Medium: CI installs an ad-hoc dependency set

- `.github/workflows/ci.yml` installs a manually enumerated set of packages rather than installing a clearly defined backend/test dependency set from a tracked lock or requirements file.
- It injects placeholder provider credentials globally for tests.
- This can allow local, CI, and container environments to drift. Dependency groups and test configuration should be made reproducible without using fake credentials as a substitute for dependency/configuration design.

### Medium: the synthetic volume test is not an independent validation of the implementation

- `tests/unit/vision/test_volume_math.py` constructs a synthetic scene and compares the implementation against a second computation using the same ray-volume equation.
- This is useful for checking internal consistency, but it does not independently establish correctness against a separately derived analytic reference or measured ground truth.
- It also assumes a flat visible support plane, so it does not validate the broader product capture constraints.

### Documentation consistency issue

- `README.md` and `docs/README.md` link to `docs/DEPLOYMENT.md`, `docs/KNOWN_LIMITATIONS.md`, and `docs/architecture/component-design.md`.
- Those paths were not present in the tracked `main` tree at the audited commit.
- Documentation links must be reconciled with actual tracked files.

## 3. Existing strengths worth preserving

- Backend and inference responsibilities are separated at the service boundary.
- Inference orchestration, model loading, and volume mathematics have distinct modules.
- The repository has unit, API, contract, regression, security, and end-to-end test directories.
- Architecture notes and ADRs capture important decisions, including removal of an unsafe absolute-scale fallback.
- Benchmark and diagnostic scripts provide a foundation for evidence-driven model and geometry evaluation.
- Docker and GitHub Actions configuration already exist and should be corrected and verified, not replaced without cause.

## 4. Proposed remediation order

1. Establish a reproducible baseline: dependency installation, import checks, test collection, and CI status.
2. Fix build and documentation integrity: reconcile Docker COPY inputs and broken documentation links.
3. Define service contracts and configuration boundaries: settings validation, auth behavior, error envelopes, CORS, and provider configuration.
4. Harden upload handling: bounded reads, actual image decoding, supported-format policy, and cleanup guarantees.
5. Correct request execution/concurrency: isolate blocking model/provider work and establish explicit capacity limits.
6. Resolve the authoritative volume contract: separate relative geometry, metric scale evidence, volume estimate, uncertainty, and validity status.
7. Build independent evaluation: analytically derived fixtures, measured real-world ground truth, dataset protocol, and error reporting.
8. Improve observability and production deployment only after the preceding contracts are stable.

## 5. Engineering constraints

- Do not commit secrets or credentials.
- Do not use guessed physical dimensions or silent numeric defaults as metric scale.
- Do not mark a volume as physically valid merely because a formula returned a positive number.
- Do not introduce arbitrary thresholds, retry counts, worker counts, or resource limits without a documented rationale and configuration contract.
- Preserve existing behavior unless a change is justified, tested, and documented.
- Do not merge this work into `main` without explicit repository-owner approval.

## 6. Verification status

This baseline was produced by reading repository metadata and source files through GitHub. No local checkout, Docker build, pytest run, model inference, live provider request, or real-world measurement was executed as part of this audit. Those checks remain pending and must be reported separately when run.
