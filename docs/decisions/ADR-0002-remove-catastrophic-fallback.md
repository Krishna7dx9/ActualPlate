# ADR-0002: Remove Catastrophic Absolute-Scale Fallback

## Status
Accepted

## Context
When the VLM label (e.g., "corn on the cob") failed to match the DINO label (e.g., "corn"), the API fell back to a raw point-cloud volume calculation. Because MoGe-2 absolute scale is hallucinated, this returned mathematically impossible volumes up to 158 Billion cm³.

## Decision
We removed the catastrophic absolute-scale fallback and retained controlled label aliasing. However, a VLM-derived diameter/thickness heuristic remained in the API and was incorrectly marked valid. ADR-0003 supersedes that behavior: VLM estimates are diagnostic only, and authoritative volume remains null until metric scale evidence is independently validated.

## Consequences
- **Pros:** The service must prefer a null physical measurement to a fabricated or unvalidated one.
- **Cons:** Physical volume and dependent nutrition coverage remain limited until a validated metric scale source is available.