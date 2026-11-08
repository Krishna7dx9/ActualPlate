# System Overview

ActualPlate is a containerized AI pipeline that estimates food volume, weight, and nutrition from a single monocular RGB image. 

## Problem Statement
Traditional dietary tracking requires manual data entry. Computer vision can automate this, but 3D volume estimation from a 2D image suffers from absolute scale ambiguity (a small plate close to the camera looks identical to a large plate far away).

## Architecture Topology
We enforce a strict microservice split to ensure GPU-bound ML inference does not block fast HTTP API routing.

```mermaid
graph TD
    Client[Client App] -->|POST /upload-image| API[Backend API :8000]
    API -->|Proxy| Worker[Vision Worker :8001]
    
    subgraph Vision Pipeline
        Worker -->|1. Prompt| Gemma[Gemma VLM: Food ID & Scale]
        Worker -->|2. Detect| DINO[Grounding DINO]
        Worker -->|3. Segment| SAM2[SAM2 Masks]
        Worker -->|4. Depth| MoGe2[MoGe-2 3D Depth]
    end
    
    Worker -->|5. Volume Math| API
    API -->|6. Nutrition Query| FatSecret[FatSecret API]
    API -->|7. Final JSON| Client