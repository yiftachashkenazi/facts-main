# Fact Checking API

A FastAPI-based fact-checking system that analyzes text and provides verification results.

## Features

- Text analysis and fact-checking
- Author information processing
- RSS feed integration
- Wikipedia data correlation
- Comprehensive event analysis

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- Gemini API key

## Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd fact-checking
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Gemini API key
```

4. Run the development server:
```bash
uvicorn api:app --reload
```

## Docker Deployment

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. Access the API at `http://localhost:8000`

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /fact-check` - Perform fact-checking analysis
- `POST /fact-check-json` - Alternative endpoint accepting JSON input
- `POST /update-vectors` - Update RSS vectors

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Production Deployment

1. Build the Docker image:
```bash
docker build -t fact-checking-api .
```

2. Run the container:
```bash
docker run -d -p 8000:8000 --env-file .env fact-checking-api
```

## Environment Variables

- `GEMINI_API_KEY` - Your Gemini API key
- Additional environment variables can be added to `.env`

## Project Structure

```
.
├── api.py              # FastAPI application
├── api_wrapper.py      # API wrapper functions
├── main1.py           # Core processing logic
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── data/             # Data directory
│   └── vectors/      # Vector storage
├── outputs/          # Output directory
└── README.md         # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here] 