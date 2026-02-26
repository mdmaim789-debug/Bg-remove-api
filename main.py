"""
Production-ready Background Removal API using FastAPI and rembg
Single file implementation - Optimized for Render Free Tier
"""

import io
import os
import logging
from typing import Set

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import rembg
import uvicorn
from pydantic_settings import BaseSettings

# ==================== CONFIGURATION ====================

class Settings(BaseSettings):
    """Application settings"""
    PROJECT_NAME: str = "Background Removal API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # File upload settings - Reduced for free tier
    MAX_FILE_SIZE: int = 5 * 1024 * 1024  # 5MB for free tier
    ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS settings
    BACKEND_CORS_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== IMAGE PROCESSOR ====================

class ImageProcessor:
    """Handles image processing operations"""
    
    def __init__(self):
        """Initialize the image processor with rembg session"""
        try:
            # Use CPU-only mode for Render
            os.environ["REMBG_USE_CPU"] = "1"
            self.rembg_session = rembg.new_session(model_name="u2net")
            logger.info("ImageProcessor initialized with rembg session (CPU mode)")
        except Exception as e:
            logger.error(f"Failed to initialize rembg session: {str(e)}")
            raise
    
    async def remove_background(self, image_bytes: bytes) -> bytes:
        """
        Remove background from image and return processed image bytes
        """
        try:
            # Open image with Pillow
            input_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
            
            # Remove background using rembg
            output_image = rembg.remove(input_image, session=self.rembg_session)
            
            # Convert to PNG and get bytes
            img_byte_arr = io.BytesIO()
            output_image.save(img_byte_arr, format='PNG', optimize=True)
            img_byte_arr.seek(0)
            
            return img_byte_arr.getvalue()
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing image: {str(e)}"
            )
    
    async def close(self):
        """Clean up resources"""
        if self.rembg_session:
            logger.info("Cleaning up rembg session")
            pass

# Create global image processor instance
image_processor = ImageProcessor()

# ==================== VALIDATORS ====================

async def validate_image_file(file: UploadFile) -> None:
    """
    Validate uploaded image file for size and type
    """
    # Check file size
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    await file.seek(0)
    
    if file_size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.MAX_FILE_SIZE / (1024 * 1024)}MB"
        )
    
    # Check file extension
    if file.filename:
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
    
    # Validate it's actually an image
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File is not a valid image"
        )

# ==================== API ENDPOINTS ====================

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Background Removal API using FastAPI and rembg",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "status": "online"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return JSONResponse(
        content={
            "status": "healthy", 
            "version": settings.VERSION,
            "service": "background-removal-api"
        },
        status_code=200
    )

@app.post(f"{settings.API_V1_STR}/remove-bg/", 
         response_class=StreamingResponse,
         summary="Remove background from image",
         description="Upload an image and get back a PNG with background removed")
async def remove_background(
    file: UploadFile = File(..., description="Image file to process (jpg, jpeg, png, webp, bmp) - max 5MB")
):
    """
    Remove background from uploaded image
    
    - **file**: Image file (jpg, jpeg, png, webp, bmp) - max 5MB
    - Returns: PNG image with transparent background
    """
    try:
        # Validate the uploaded file
        await validate_image_file(file)
        
        # Read file content
        contents = await file.read()
        logger.info(f"Processing file: {file.filename}, size: {len(contents)} bytes")
        
        # Process image (remove background)
        processed_image_bytes = await image_processor.remove_background(contents)
        
        # Generate filename for download
        original_filename = file.filename or "image"
        base_name = os.path.splitext(original_filename)[0]
        download_filename = f"no_bg_{base_name}.png"
        
        logger.info(f"Successfully processed: {file.filename}")
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(processed_image_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename={download_filename}",
                "Content-Length": str(len(processed_image_bytes))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

# ==================== EVENT HANDLERS ====================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("=" * 50)
    logger.info("Starting up Background Removal API...")
    logger.info(f"Version: {settings.VERSION}")
    logger.info(f"Max file size: {settings.MAX_FILE_SIZE / (1024 * 1024)}MB")
    logger.info(f"Allowed extensions: {settings.ALLOWED_EXTENSIONS}")
    logger.info("API is ready to accept requests")
    logger.info("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down Background Removal API...")
    await image_processor.close()
    logger.info("Shutdown complete")

# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=port,
        reload=settings.DEBUG
    )
