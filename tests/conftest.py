import os

# Test-only credentials and local service endpoint. Never use these values
# outside the isolated test process.
os.environ["VISION_API_TOKEN"] = "test-token"
os.environ["FATSECRET_CLIENT_ID"] = "test-id"
os.environ["FATSECRET_CLIENT_SECRET"] = "test-secret"
os.environ["VISION_INFERENCE_ENDPOINT"] = "http://localhost:8001"
