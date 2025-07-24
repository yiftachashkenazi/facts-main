#!/bin/bash

# Docker build script with multiple options
set -e

echo "🐳 Fact-Checking API Docker Build Script"
echo "========================================"

# Function to build with different strategies
build_image() {
    local dockerfile=$1
    local tag=$2
    local description=$3
    
    echo ""
    echo "🔨 Building with $description..."
    echo "Using: $dockerfile"
    echo "Tag: $tag"
    
    if docker build -f "$dockerfile" -t "$tag" .; then
        echo "✅ Successfully built $tag"
        return 0
    else
        echo "❌ Failed to build $tag"
        return 1
    fi
}

# Option 1: Try lightweight build first (fastest)
echo "Option 1: Lightweight build (optimized dependencies)"
if build_image "Dockerfile.light" "fact-checker:light" "lightweight Docker build"; then
    echo "🎉 Lightweight build successful!"
    echo "To run: docker run -p 8000:8000 fact-checker:light"
    exit 0
fi

# Option 2: Try standard build
echo "Option 2: Standard build"
if build_image "Dockerfile" "fact-checker:standard" "standard Docker build"; then
    echo "🎉 Standard build successful!"
    echo "To run: docker run -p 8000:8000 fact-checker:standard"
    exit 0
fi

# Option 3: Try multi-stage build (if others fail)
echo "Option 3: Multi-stage build (most robust)"
if build_image "Dockerfile.multistage" "fact-checker:multistage" "multi-stage Docker build"; then
    echo "🎉 Multi-stage build successful!"
    echo "To run: docker run -p 8000:8000 fact-checker:multistage"
    exit 0
fi

echo ""
echo "❌ All build strategies failed!"
echo "Please check the Docker troubleshooting guide: DOCKER_TROUBLESHOOTING.md"
exit 1 