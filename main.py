from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from graph import MultimodalQAGraph
from typing import Optional
from image_utils import process_image, validate_image_format

# Load environment variables from .env file
load_dotenv()

# Get API keys and AWS credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_REGION = os.getenv("AWS_REGION", "us-west-1")

if not all([OPENAI_API_KEY, AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_SESSION_TOKEN]):
    raise ValueError("Required API keys or AWS credentials not found in environment variables")

app = FastAPI(
    title="Multimodal QnA Agent (Server)",
    description="A FastAPI service that processes questions with optional images using OpenAI and AWS Bedrock",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize graph with OpenAI and AWS credentials
graph = MultimodalQAGraph(
    openai_api_key=OPENAI_API_KEY,
    aws_access_key=AWS_ACCESS_KEY,
    aws_secret_key=AWS_SECRET_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
    aws_region=AWS_REGION
)
chain = graph.build()

@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    """
    Process a question with an optional image.
    Supports PNG and JPG/JPEG image formats.
    
    Args:
        question (str): The question to be answered
        image (UploadFile, optional): An image file to analyze (PNG or JPG/JPEG)
    """
    try:
        # Validate image format if provided
        if image:
            if not validate_image_format(image.filename):
                raise HTTPException(
                    status_code=400,
                    detail="Only PNG and JPG/JPEG images are supported"
                )
            
            # Process image
            image_data = process_image(image)
            
            # Debug prints
            print(f"Question : {question}")
            print(f"Image file name: {image.filename}")
            print(f"Base64 string starts with: {image_data[:50]}...")

        else:
            image_data = None

        # Run the chain
        result = chain.invoke({
            "question": question,
            "image": image_data,
            "answer": None,
            "diagram": None
        })

        return JSONResponse({
            "status": "success",
            "answer": result["answer"],
            "diagram": result["diagram"]
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 