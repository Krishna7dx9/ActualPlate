This document traces the exact execution path of an image through the vision pipeline.

## Stage 1: Food Recognition (Gemma)
- **Input:** Raw RGB Image.
- **Processing:** OpenRouter (Gemma) is prompted to identify the food and estimate its physical 2D diameter (`diameter_cm`) and 3D thickness (`thickness_cm`).
- **Output:** JSON array of detected foods with physical scale estimates.
- **Failure Behavior:** If the VLM returns invalid JSON, the pipeline catches the exception and returns `status: "unreliable"`.

## Stage 2: Object Detection (Grounding DINO)
- **Input:** RGB Image + Prompts from Stage 1.
- **Processing:** Zero-shot object detection generates 2D bounding boxes.
- **Output:** Bounding boxes `[x_min, y_min, x_max, y_max]`.

## Stage 3: Instance Segmentation (SAM2)
- **Input:** RGB Image + Bounding Boxes from Stage 2.
- **Processing:** SAM2 generates pixel-perfect binary masks for each detected food item.
- **Output:** 2D NumPy boolean masks (HxW).

## Stage 4: Metric Depth Estimation (MoGe-2)
- **Input:** RGB Image.
- **Processing:** MoGe-2 (MIT License) generates a relative 3D point cloud and depth map.
- **Output:** Depth map (relative Z-axis values).

## Stage 5: Volume Integration (Math Engine)
- **Input:** 2D Mask (Area) + Depth Map + VLM Thickness (`thickness_cm`).
- **Processing:** The depth map is normalized (0-1) and multiplied by the VLM's thickness estimate to force absolute scale. Volume is calculated as `integral(Area * thickness)`.
- **Output:** Predicted volume in `cm³`.