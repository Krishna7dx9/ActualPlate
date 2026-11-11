# Component Design

## Backend API

**Responsibilities**
- Accept client requests and validate request-level inputs.
- Orchestrate the vision request and nutrition enrichment.
- Convert provider failures into stable application-level outcomes.
- Return structured results to clients.

**Key areas**
- backend/app/api/routes/: HTTP endpoints and request/response handling.
- backend/app/services/pipeline.py: application orchestration.
- backend/app/nutrition/: nutrition provider integration and result formatting.

The backend should not own GPU model state. Blocking model/provider calls must not run directly on the async event loop; use bounded execution and explicit timeouts.

## Vision service

**Responsibilities**
- Run food detection, instance segmentation, and monocular geometry.
- Own model loading and the lifecycle of model instances.
- Return typed, serializable detection and geometry results.
- Separate diagnostic estimates from validated physical measurements.

**Key areas**
- vision_service/recognition/: food-label recognition and optional visual priors.
- vision_service/detection/: backend client for the remote vision API and detection integration.
- vision_service/inference/api.py: HTTP boundary for inference requests.
- vision_service/inference/inference.py: model orchestration and per-food outputs.
- vision_service/inference/models.py: lazy model loading.
- vision_service/inference/volume.py: standalone ray/plane volume utility with explicit metric-depth and support-plane assumptions.

## Data flow

1. Backend validates the upload and reads bounded image bytes.
2. Food recognition proposes labels and optional visual priors.
3. Grounding DINO proposes 2D regions.
4. SAM2 predicts per-instance masks.
5. MoGe provides monocular geometry/depth and predicted intrinsics.
6. A physical volume may be reported only if a validated metric-scale source is available. A heuristic derived from VLM dimensions is diagnostic, not authoritative.
7. Nutrition enrichment requires a defensible mass estimate; provider failure must be reported separately.

## Contracts and boundaries

- HTTP request/response schemas belong at service boundaries.
- Domain calculations should be testable without starting FastAPI or loading GPU models.
- Runtime settings should come from validated configuration, not environment-specific literals scattered across modules.
- Error messages returned to clients must be stable and must not contain raw provider exceptions or secrets.
- Algorithm parameters should be documented and tested; do not change thresholds merely to improve a single benchmark.
- Any physical unit must be explicit. Relative depth values must never be labeled as meters.

## Current architecture caveat

The codebase contains both a normalized-depth/VLM-dimension heuristic and a separate support-plane geometry utility. The active production path must be made explicit, and these implementations must not be treated as interchangeable. The engineering roadmap tracks this as a critical correctness task.
