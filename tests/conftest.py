import os

# Set dummy environment variables for testing BEFORE any imports happen.
# This prevents the app from crashing when it looks for tokens during test collection.
os.environ["VISION_API_TOKEN"] = "test-token"
os.environ["FATSECRET_CLIENT_ID"] = "test-id"
os.environ["FATSECRET_CLIENT_SECRET"] = "test-secret"
os.environ["VISION_INFERENCE_ENDPOINT"] = "http://localhost:8001"
os.environ["OPENROUTER_API_KEY"] = "test-only-openrouter-key"
os.environ["MAX_UPLOAD_BYTES"] = "10485760"
os.environ["MAX_REQUEST_BODY_BYTES"] = "12582912"
os.environ["MAX_IMAGE_PIXELS"] = "40000000"
os.environ["APP_ENV"] = "development"
os.environ["OPENROUTER_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["OPENROUTER_MODEL"] = "google/gemma-3-27b-it"
os.environ["OPENROUTER_MAX_TOKENS"] = "800"
os.environ["GROUNDING_DINO_MODEL_ID"] = "IDEA-Research/grounding-dino-base"
os.environ["SAM2_MODEL_ID"] = "facebook/sam2-hiera-large"
os.environ["MOGE_MODEL_ID"] = "Ruicheng/moge-2-vitl-normal"
os.environ["BOX_THRESHOLD"] = "0.35"
os.environ["TEXT_THRESHOLD"] = "0.25"
os.environ["SHOW_VISUALIZATIONS"] = "false"
os.environ["VISION_MAX_ARRAY_SIDE"] = "512"
os.environ["VISION_REQUEST_TIMEOUT_SECONDS"] = "300"
