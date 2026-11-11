# Image input and prompt-injection safety

## Upload pipeline

The backend enforces byte and pixel limits, decodes the payload rather than trusting the client MIME type, rejects unsupported/animated formats, applies EXIF orientation to the pixels, and re-encodes JPEG/PNG/WebP data before inference. The sanitized payload is rewound into the upload object, so downstream inference does not receive the original byte stream.

Re-encoding removes EXIF/XMP/comment/ICC payloads and trailing non-image bytes that are not part of the decoded pixels. It is not a mathematical guarantee against pixel-level steganography; hidden information encoded into image pixels can survive re-encoding. Do not claim otherwise.

The sanitized output must still fit the configured upload byte limit. Images whose encoded output grows beyond that limit are rejected rather than bypassing the request budget.

## Recognition prompt trust boundary

- The system prompt is a versioned asset, not an inline Python string.
- Image text, packaging, labels, signs, watermarks, and QR codes are explicitly treated as untrusted visual data, never as instructions.
- The recognition model and provider endpoint are configured through environment variables; the client is created lazily so missing provider credentials do not break unrelated imports or health checks.
- Density and dimensions are nullable. Boolean, non-finite, zero, and negative estimates become `null`; no default physical dimensions are fabricated.
- Nutrition enrichment is skipped when density or volume is unavailable. The API reports complete/partial/unavailable nutrition status and safe error codes instead of silently swallowing all failures.

Prompt instructions are defense in depth, not a guarantee that a vision model cannot be influenced by adversarial text in an image. The current pipeline does not OCR-extract and execute image text; dedicated text-region detection/redaction would require separate evaluation because masking text can also erase legitimate food labels or visual evidence.

## Provider configuration

- `OPENROUTER_API_KEY`: secret; do not commit it or expose it to the frontend.
- `OPENROUTER_BASE_URL`: optional absolute provider URL; production must use HTTPS.
- `FOOD_RECOGNITION_MODEL`: required model identifier; no model ID is silently hardcoded in the recognition code.
- `FOOD_RECOGNITION_TIMEOUT_SECONDS`: 1–300 seconds, default 60.
- `FOOD_RECOGNITION_MAX_TOKENS`: 1–4000, default 800.
- `FOOD_RECOGNITION_TEMPERATURE`: 0–2, default 0.

Missing credentials/model configuration are reported when recognition is requested, not during module import. Production deployment must provision these settings through a secret/configuration manager.

## Verification boundaries

Unit tests cover supported decoding, byte/pixel bounds, EXIF orientation removal, animated-image rejection, prompt trust instructions, MIME correctness, and nullable measurements. They do not prove resistance to every steganography technique, adversarial prompt, or real-world recognition failure. A labeled adversarial-image evaluation set remains a release gate.
