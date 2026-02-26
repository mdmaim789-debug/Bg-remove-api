"""
Production-ready Background Removal API using FastAPI and rembg
Complete fixed version with NumPy 1.x compatibility
"""

# ==================== FIX NUMPY ISSUE AT THE VERY TOP ====================
import os
import sys

# Force NumPy 1.x behavior BEFORE any other imports
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent thread conflicts
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Prevent thread conflicts

# ==================== STANDARD IMPORTS ====================
import io
import logging
from typing import Set, Optional
from datetime import datetime

# ==================== THIRD-PARTY IMPORTS ====================
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exception_handlers import http_exception_handler
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
    
    # File upload settings - Reduced for free tier memory limits
    MAX_FILE_SIZE: int = 3 * 1024 * 1024  # 3MB max for free tier
    ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ALLOWED_MIME_TYPES: Set[str] = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS settings
    BACKEND_CORS_ORIGINS: list = ["*"]
    
    # Model settings
    REMBG_MODEL: str = "u2netp"  # Use smaller model for free tier (u2netp instead of u2net)
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', mode='a') if os.access('.', os.W_OK) else logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==================== CUSTOM EXCEPTIONS ====================

class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass

class FileValidationError(Exception):
    """Custom exception for file validation errors"""
    pass

# ==================== IMAGE PROCESSOR ====================

class ImageProcessor:
    """Handles image processing operations with proper error handling"""
    
    def __init__(self):
        """Initialize the image processor with rembg session"""
        self.rembg_session = None
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize rembg session with error handling"""
        try:
            # Set CPU mode explicitly
            os.environ["REMBG_USE_CPU"] = "1"
            
            # Use smaller model for free tier
            logger.info(f"Initializing rembg with model: {settings.REMBG_MODEL}")
            self.rembg_session = rembg.new_session(model_name=settings.REMBG_MODEL)
            logger.info(f"✓ ImageProcessor initialized with {settings.REMBG_MODEL} model")
        except ImportError as e:
            if "numpy" in str(e).lower():
                logger.error("NumPy version mismatch detected!")
                logger.error("Please ensure numpy 1.x is installed: pip install numpy==1.24.3")
                raise ImportError("NumPy version mismatch. Please install numpy==1.24.3")
            else:
                logger.error(f"Failed to initialize rembg session: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Failed to initialize rembg session: {str(e)}")
            raise
    
    async def remove_background(self, image_bytes: bytes) -> bytes:
        """
        Remove background from image and return processed image bytes
        """
        if not self.rembg_session:
            self._initialize_session()
        
        try:
            # Open image with Pillow
            input_image = Image.open(io.BytesIO(image_bytes))
            
            # Log image details
            logger.info(f"Image mode: {input_image.mode}, size: {input_image.size}")
            
            # Convert to RGB if necessary (rembg works best with RGB)
            if input_image.mode not in ['RGB', 'RGBA']:
                input_image = input_image.convert('RGB')
            
            # Remove background using rembg
            output_image = rembg.remove(
                input_image, 
                session=self.rembg_session,
                alpha_matting=True,  # Better edge detection
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10
            )
            
            # Convert to PNG and get bytes
            img_byte_arr = io.BytesIO()
            
            # Save with optimization
            output_image.save(
                img_byte_arr, 
                format='PNG', 
                optimize=True,
                compress_level=6  # Balance between size and speed
            )
            img_byte_arr.seek(0)
            
            processed_size = len(img_byte_arr.getvalue())
            logger.info(f"Processed image size: {processed_size} bytes")
            
            return img_byte_arr.getvalue()
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise ImageProcessingError(f"Failed to process image: {str(e)}")
    
    async def close(self):
        """Clean up resources"""
        if self.rembg_session:
            logger.info("Cleaning up rembg session")
            # Force garbage collection
            import gc
            gc.collect()

# Create global image processor instance
image_processor = ImageProcessor()

# ==================== VALIDATORS ====================

async def validate_image_file(file: UploadFile) -> None:
    """
    Validate uploaded image file for size, type, and integrity
    """
    # Check if file exists
    if not file:
        raise FileValidationError("No file uploaded")
    
    # Check filename
    if not file.filename:
        raise FileValidationError("File has no filename")
    
    # Check file size
    try:
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        await file.seek(0)
        
        if file_size > settings.MAX_FILE_SIZE:
            max_size_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
            raise FileValidationError(f"File too large. Max size: {max_size_mb:.1f}MB")
        
        if file_size == 0:
            raise FileValidationError("File is empty")
            
    except Exception as e:
        raise FileValidationError(f"Error reading file: {str(e)}")
    
    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Invalid file type. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    # Check MIME type
    if file.content_type:
        if file.content_type not in settings.ALLOWED_MIME_TYPES:
            raise FileValidationError(
                f"Invalid MIME type. Allowed: {', '.join(settings.ALLOWED_MIME_TYPES)}"
            )
    else:
        # If no content type, try to detect from extension
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise FileValidationError("Could not determine file type")

# ==================== API ENDPOINTS ====================

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-ready Background Removal API using FastAPI and rembg",
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

# ==================== EXCEPTION HANDLERS ====================

@app.exception_handler(FileValidationError)
async def file_validation_exception_handler(request: Request, exc: FileValidationError):
    """Handle file validation errors"""
    return JSONResponse(
        status_code=400,
        content={
            "error": "File Validation Error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(ImageProcessingError)
async def image_processing_exception_handler(request: Request, exc: ImageProcessingError):
    """Handle image processing errors"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Image Processing Error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# ==================== HEALTH AND INFO ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return JSONResponse(
        content={
            "status": "healthy",
            "version": settings.VERSION,
            "service": "background-removal-api",
            "timestamp": datetime.now().isoformat(),
            "memory_usage": f"{get_memory_usage():.1f}MB" if has_memory_usage() else "unknown"
        },
        status_code=200
    )

def has_memory_usage():
    """Check if psutil is available"""
    try:
        import psutil
        return True
    except ImportError:
        return False

def get_memory_usage():
    """Get current memory usage"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

@app.get("/debug")
async def debug_info():
    """Debug endpoint - only works in DEBUG mode"""
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Not found")
    
    return {
        "numpy_version": get_numpy_version(),
        "pillow_version": Image.__version__,
        "rembg_version": rembg.__version__,
        "max_file_size_mb": settings.MAX_FILE_SIZE / (1024 * 1024),
        "model": settings.REMBG_MODEL,
        "python_version": sys.version
    }

def get_numpy_version():
    """Get numpy version safely"""
    try:
        import numpy
        return numpy.__version__
    except:
        return "not installed"

# ==================== MAIN BACKGROUND REMOVAL ENDPOINT ====================

@app.post(f"{settings.API_V1_STR}/remove-bg/", 
         response_class=StreamingResponse,
         summary="Remove background from image",
         description="Upload an image and get back a PNG with background removed")
async def remove_background(
    file: UploadFile = File(..., description="Image file (JPG, PNG, WEBP, BMP) - max 3MB")
):
    """
    Remove background from uploaded image
    
    - **file**: Image file (jpg, jpeg, png, webp, bmp) - max 3MB
    - Returns: PNG image with transparent background
    
    **Supported Formats:** JPG, JPEG, PNG, WEBP, BMP
    **Max File Size:** 3MB
    **Output:** PNG with transparency
    """
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    
    try:
        logger.info(f"[{request_id}] Processing request for file: {file.filename}")
        
        # Validate the uploaded file
        await validate_image_file(file)
        
        # Read file content
        contents = await file.read()
        logger.info(f"[{request_id}] File size: {len(contents)} bytes")
        
        # Process image (remove background)
        processed_image_bytes = await image_processor.remove_background(contents)
        
        # Generate safe filename for download
        original_filename = file.filename or "image"
        base_name = os.path.splitext(original_filename)[0]
        # Remove any potentially unsafe characters
        safe_base_name = "".join(c for c in base_name if c.isalnum() or c in "._- ")
        download_filename = f"no_bg_{safe_base_name}.png"
        
        logger.info(f"[{request_id}] Successfully processed: {file.filename}")
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(processed_image_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename={download_filename}",
                "Content-Length": str(len(processed_image_bytes)),
                "X-Request-ID": request_id
            }
        )
        
    except (FileValidationError, ImageProcessingError, HTTPException) as e:
        # Re-raise these as they're handled by exception handlers
        logger.warning(f"[{request_id}] {type(e).__name__}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred. Request ID: {request_id}"
        )

# ==================== EVENT HANDLERS ====================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("=" * 60)
    logger.info("🚀 Starting Background Removal API...")
    logger.info("=" * 60)
    logger.info(f"Version: {settings.VERSION}")
    logger.info(f"Environment: {'Development' if settings.DEBUG else 'Production'}")
    logger.info(f"Max file size: {settings.MAX_FILE_SIZE / (1024 * 1026):.1f}MB")
    logger.info(f"Allowed extensions: {', '.join(settings.ALLOWED_EXTENSIONS)}")
    logger.info(f"Model: {settings.REMBG_MODEL}")
    
    # Check numpy version
    try:
        import numpy
        logger.info(f"NumPy version: {numpy.__version__}")
        if numpy.__version__.startswith('2.'):
            logger.warning("⚠️ NumPy 2.x detected - This may cause issues!")
    except ImportError:
        logger.error("❌ NumPy not installed!")
    
    logger.info("=" * 60)
    logger.info("✅ API is ready to accept requests")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("🛑 Shutting down Background Removal API...")
    await image_processor.close()
    logger.info("✅ Shutdown complete")

# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=port,
        reload=settings.DEBUG,
        log_level="info"
    )
