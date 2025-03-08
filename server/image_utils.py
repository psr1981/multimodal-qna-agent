from fastapi import UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import base64
import sys

def process_image(image_file: UploadFile) -> str:
    """
    Process uploaded image file and convert to base64.
    Supports PNG and JPG/JPEG formats.
    
    Args:
        image_file (UploadFile): Uploaded image file
        
    Returns:
        str: Base64 encoded image with data URL format
    """
    try:
        contents = image_file.file.read()
        img = Image.open(BytesIO(contents))
        
        # Convert image to RGB if it's in RGBA mode
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Keep original format
        file_ext = image_file.filename.lower().split('.')[-1]
        img_format = 'PNG' if file_ext == 'png' else 'JPEG'
        
        
        # Save image to buffer
        buffered = BytesIO()
        if img_format == 'PNG':
            img.save(buffered, format='PNG', optimize=True)
        else:
            img.save(buffered, format='JPEG', quality=95, optimize=True)
        

        buffered.seek(0)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Return the base64 string with correct mime type
        mime_type = 'png' if img_format == 'PNG' else 'jpeg'

        return f"data:image/{mime_type};base64,{img_str}"
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )
    finally:
        image_file.file.close()

def validate_image_format(filename: str) -> bool:
    """
    Validate if the image format is supported (PNG or JPG/JPEG).
    
    Args:
        filename (str): Name of the uploaded file
        
    Returns:
        bool: True if format is supported, False otherwise
    """
    allowed_extensions = ['png', 'jpg', 'jpeg']
    file_ext = filename.lower().split('.')[-1]
    return file_ext in allowed_extensions

def verify_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        print("Image is valid")
        return True
    except Exception as e:
        print(f"Image is invalid: {e}")
        return False 