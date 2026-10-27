# Baseline Benchmark (v1.0.0)
* **Date:** 27-10-2026
* **Architecture:** Depth Pro + VLM Plateau Approximation (Hackathon MVP)
* **Dataset:** SimpleFood45 (N=420 Images)
* **Valid Predictions:** 392
* **Baseline End-to-End Volume MAPE:** **123.67%**

## Known Failure Modes
1. Complex multi-food plates result in extreme volume underestimation (e.g., 19cm³ for a full meal).
2. Gemma hallucinates thickness on ambiguous shapes.
3. VLM detection fails on certain food classes (resulting in ~6% detection failure rate).