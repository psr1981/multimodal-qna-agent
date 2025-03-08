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

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

app = FastAPI(
    title="Multimodal QnA Agent (Server)",
    description="A FastAPI service that processes questions with optional images using OpenAI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize graph with OpenAI API key
graph = MultimodalQAGraph(api_key=OPENAI_API_KEY)
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
            #print(f"Image filename: {image.filename}")
            #print(f"Image content type: {image.content_type}")
           # print(f"Base64 string starts with: {image_data[:50]}...")

        else:
            image_data = None

         # Run the chain
        result = chain.invoke({
            "question": question,
            "image": image_data,
            "chat_history": []
        })

        return JSONResponse({
            "status": "success",
            "answer": result["answer"]
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