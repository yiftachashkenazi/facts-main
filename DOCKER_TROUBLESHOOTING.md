# Docker Build Troubleshooting Guide

## 🐳 Common Docker Build Issues and Solutions

### Issue: `pip install` fails with exit code 1

This is typically caused by:
1. **Insufficient memory** - ML packages like PyTorch require significant memory
2. **Missing system dependencies** - Some packages need build tools
3. **Network timeouts** - Large packages may timeout during download
4. **Platform compatibility** - Some packages don't have pre-built wheels

## 🔧 Solutions

### Option 1: Use the Lightweight Build (Recommended for Testing)

```bash
# Build with minimal dependencies (fastest)
docker build -f Dockerfile.light -t fact-checker:light .
docker run -p 8000:8000 fact-checker:light
```

**Pros:**
- Fast build time
- Smaller image size
- Uses SimpleSimilarityModel instead of heavy ML models

**Cons:**
- Limited ML capabilities
- No sentence transformers

### Option 2: Use the Build Script

```bash
# Try multiple build strategies automatically
./build-docker.sh
```

This script will try different approaches in order of likelihood to succeed.

### Option 3: Increase Docker Resources

1. **Open Docker Desktop**
2. **Go to Settings > Resources**
3. **Increase Memory to 4GB+**
4. **Increase Swap to 2GB+**
5. **Apply & Restart Docker**

Then try:
```bash
docker build --memory=4g --memory-swap=6g -t fact-checker:full .
```

### Option 4: Manual Step-by-Step Build

If all else fails, build dependencies manually:

```bash
# Build base image
docker build -f Dockerfile.light -t fact-checker:base .

# Run container and install additional packages
docker run -it --name temp-container fact-checker:base bash

# Inside container:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.46.3
pip install sentence-transformers==3.3.1

# Commit the container
docker commit temp-container fact-checker:full
docker rm temp-container
```

## 🚀 Alternative Deployment Options

### Option A: Use Docker Compose with Build Args

```yaml
# docker-compose.yml
version: '3.8'
services:
  fact-checker:
    build:
      context: .
      dockerfile: Dockerfile.light
      args:
        - BUILDKIT_INLINE_CACHE=1
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./data:/app/data
```

### Option B: Use Pre-built Base Images

Create a Dockerfile that uses a pre-built ML image:

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements-minimal.txt .
RUN pip install -r requirements-minimal.txt
COPY . .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔍 Debugging Build Issues

### Check Docker Resources
```bash
docker system df
docker system info
```

### Build with Verbose Output
```bash
docker build --progress=plain --no-cache -t fact-checker .
```

### Check Available Space
```bash
df -h
docker system prune -f
```

## 📦 Available Docker Images

After successful build, you'll have these options:

1. **`fact-checker:light`** - Minimal dependencies, fast startup
2. **`fact-checker:multistage`** - Optimized multi-stage build
3. **`fact-checker:optimized`** - Full ML capabilities
4. **`fact-checker:full`** - Complete build with all features

## 🏃‍♂️ Quick Start Commands

```bash
# Option 1: Lightweight (recommended for testing)
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key fact-checker:light

# Option 2: Full features
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key fact-checker:full

# Option 3: With volume mounting
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e GEMINI_API_KEY=your_key \
  fact-checker:light
```

## 🌐 Testing the Container

Once running, test the API:

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs

# Test fact-check
curl -X POST "http://localhost:8000/fact-check" \
  -H "Content-Type: application/json" \
  -d '{"text": "Test message", "author_name": "Test Author"}'
```

## 🆘 Still Having Issues?

1. **Check Docker version**: `docker --version` (requires 20.10+)
2. **Check available disk space**: Ensure 10GB+ free space
3. **Try building on a different machine** with more resources
4. **Use the native Python installation** instead of Docker for development
5. **Contact support** with the specific error message

## 💡 Performance Tips

- Use `.dockerignore` to exclude unnecessary files
- Build during off-peak hours for better network speeds
- Consider using Docker BuildKit for faster builds
- Use multi-stage builds to reduce final image size
- Cache dependencies by copying requirements first 