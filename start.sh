#!/bin/bash

# Start the Vision Worker in the background
echo "Starting Vision Worker on port 8001..."
uvicorn vision_service.inference.api:app --host 0.0.0.0 --port 8001 &

# Start the Main Backend in the foreground
echo "Starting Main Backend on port 8000..."
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?