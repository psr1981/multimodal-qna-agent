from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv
from graph import MultimodalQAGraph
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

app = FastAPI(
    title="Multimodal QnA Agent",
    description="A FastAPI service that processes questions with optional images using GPT-4-Vision",
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
    
    Args:
        question (str): The question to be answered
        image (UploadFile, optional): An image file to analyze
    """
    try:
        # Convert image to base64 if provided
        image_data = None
        if image:
            contents = await image.read()
            img = Image.open(BytesIO(contents))
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            image_data = base64.b64encode(buffered.getvalue()).decode()

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