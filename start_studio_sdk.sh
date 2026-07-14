#!/bin/bash
# start-studio.sh

# Default configuration
IMAGE_NAME="genrobot/matrix-studio:0.2.5"
CONTAINER_NAME="matrix-studio"
CURRENT_DIR=$(pwd)

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --image-name IMAGE_NAME  Set Docker image name (default: genrobot/matrix-studio:0.2.5)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

STUDIO_WORKSPACE=${STUDIO_WORKSPACE:-"/app/data"}

echo "=== Starting Matrix Studio Container (Interactive Mode) ==="
echo "Image: $IMAGE_NAME"
echo "Container Name: $CONTAINER_NAME"
echo "Host Directory: $CURRENT_DIR"
echo "Container Directory: /app"

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "❌ Image not found: $IMAGE_NAME"
    echo "Please pull the image first: docker pull $IMAGE_NAME"
    exit 1
fi

echo "Cleaning up old container..."
docker stop "$CONTAINER_NAME" 2>/dev/null
docker rm "$CONTAINER_NAME" 2>/dev/null

echo "Starting container with entrypoint..."

docker run -it \
    --name "$CONTAINER_NAME" \
    -v "$CURRENT_DIR/data:/app/data" \
    --ipc=host \
    -e STUDIO_WORKSPACE="/app/data" \
    -e HOST_CURRENT_DIR="$CURRENT_DIR" \
    --workdir /app \
    --entrypoint /bin/bash \
    "$IMAGE_NAME" \
    -c "source /entrypoint.sh; exec /bin/bash"
