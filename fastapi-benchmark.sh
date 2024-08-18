#!/bin/bash

OUTPUT_FILE="fastapi.log"

python fastapi-app.py > "$OUTPUT_FILE" 2>&1 &
PYTHON_PID=$!

cleanup() {
    echo "stopping server..."
    kill $PYTHON_PID
}

trap cleanup EXIT

# wait for the server to start
sleep 10

echo "warming up pytorch model..."
python client.py --infile "cats-image.png" --jobs 10

echo "running benchmark..."

echo "experiment 1: 10 concurrent requests"
python client.py --infile "cats-image.png" --jobs 10

echo "experiment 2: 50 concurrent requests"
python client.py --infile "cats-image.png" --jobs 50

echo "experiment 3: 100 concurrent requests"
python client.py --infile "cats-image.png" --jobs 100

echo "experiment 4: 200 concurrent requests"
python client.py --infile "cats-image.png" --jobs 200
