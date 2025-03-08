# Multimodal QnA Agent (Server)

A FastAPI-based multimodal question-answering system using OpenAI.

## Setup

1. Install uv package manager:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your OpenAI API key
# Replace 'your-api-key-here' with your actual OpenAI API key
```

## Running the Service

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

## API Usage

### Process Question with Image
```bash
curl -X POST \
  -F "question=What's in this image?" \
  -F "image=@path/to/image.jpg" \
  http://localhost:8000/ask
```

### Process Text-Only Question
```bash
curl -X POST \
  -F "question=What is the capital of France?" \
  http://localhost:8000/ask
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

- `main.py`: FastAPI application and endpoints
- `graph.py`: LangGraph workflow implementation
- `agents.py`: MultimodalAgent implementation
- `requirements.txt`: Project dependencies
