# ADR-0002: Remove Catastrophic Absolute-Scale Fallback

## Status
Accepted

## Context
When the VLM label (e.g., "corn on the cob") failed to match the DINO label (e.g., "corn"), the API fell back to a raw point-cloud volume calculation. Because MoGe-2 absolute scale is hallucinated, this returned mathematically impossible volumes up to 158 Billion cm³.

## Decision
We removed the absolute-scale fallback. We implemented a safe alias-matching dictionary (e.g., "corn" -> "corn on the cob"). If a label still fails to match, the API fails safely by returning `volume_status: "unreliable"` and `volume_cm3: None`.

## Consequences
- **Pros:** Eradicated catastrophic data errors. It is better for a B2B client to receive a null value than a wrong one.
- **Cons:** System coverage drops slightly if VLM and DINO disagree fundamentally.