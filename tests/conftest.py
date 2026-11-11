import os

# Set dummy environment variables for testing BEFORE any imports happen.
# This prevents the app from crashing when it looks for tokens during test collection.
os.environ["VISION_API_TOKEN"] = "test-token"
os.environ["FATSECRET_CLIENT_ID"] = "test-id"
os.environ["FATSECRET_CLIENT_SECRET"] = "test-secret"
os.environ["MAX_UPLOAD_BYTES"] = "10485760"
os.environ["MAX_IMAGE_PIXELS"] = "40000000"
