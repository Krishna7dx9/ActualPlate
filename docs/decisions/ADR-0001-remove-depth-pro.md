# ADR-0001: Replace Apple Depth Pro with MoGe-2

## Status
Accepted

## Context
The initial architecture relied on Apple's Depth Pro for monocular metric depth estimation. However, it required a massive 1.8GB checkpoint download, heavily bloated the Docker image, and its license was restrictive for B2B commercial use.

## Decision
We removed Apple Depth Pro entirely and replaced it with MoGe-2.

## Consequences
- **Pros:** Reduced Docker image size by ~2GB. MoGe-2 uses the MIT license and provides useful monocular geometry.
- **Cons:** MoGe-2 output does not independently establish absolute metric scale. VLM-estimated food dimensions are not metric anchors; see ADR-0003 for the required measurement contract.