# ADR-0001: Replace Apple Depth Pro with MoGe-2

## Status
Accepted

## Context
The initial architecture relied on Apple's Depth Pro for monocular metric depth estimation. However, it required a massive 1.8GB checkpoint download, heavily bloated the Docker image, and its license was restrictive for B2B commercial use.

## Decision
We removed Apple Depth Pro entirely and replaced it with MoGe-2.

## Consequences
- **Pros:** Reduced Docker image size by ~2GB. MoGe-2 uses the MIT license, clearing legal blockers for commercialization. It provides excellent relative 3D geometry.
- **Cons:** MoGe-2 does not provide absolute metric scale out-of-the-box, requiring us to rely on the VLM for physical scale anchors.