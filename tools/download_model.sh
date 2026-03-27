#!/bin/bash

# Check if ARG is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 ARG"
    exit 1
fi

# Run the docker exec command with ARG
docker compose exec ollama ollama pull "$1"
