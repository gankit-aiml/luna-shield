# main.py
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import os
import uuid
import shutil
import cv2
import torch
import torch.nn as nn # Import nn
import numpy as np
from torchvision import transforms, models # Import models
from PIL import Image
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging # Add logging

# Initialize FastAPI app
app = FastAPI(title="Luna Shield Deepfake Detector")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mount static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates for HTML rendering
templates = Jinja2Templates(directory="templates")

# Configuration (Aligned with full_pipe.py where applicable)
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results" # Although not fully used in this example
MODEL_PATH = "best_model.pth" # Assumes the model trained by full_pipe.py is here
IMG_SIZE = 224 # <<< CHANGED: Match full_pipe.py
FRAMES_TO_ANALYZE = 10 # Matches FRAMES_PER_VIDEO in full_pipe.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True) # Keep for potential future use

# --- Model Definition and Loading (Aligned with full_pipe.py) ---
logger.info(f"Using device: {DEVICE}")

try:
    # 1. Define the model architecture EXACTLY as in full_pipe.py
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if hasattr(models, 'EfficientNet_B0_Weights') else True) # Use updated weights API if available
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3), # Match dropout from full_pipe.py
        nn.Linear(num_ftrs, 2) # Output classes: 0 (real), 1 (fake)
    )
    logger.info("Model architecture defined (EfficientNet-B0).")

    # 2. Load the trained weights
    if not os.path.exists(MODEL_PATH):
         raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Make sure 'best_model.pth' from training is present.")

    # Load state dict, ensuring map_location handles CPU/GPU discrepancy
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    logger.info(f"Model weights loaded successfully from {MODEL_PATH}.")

    # 3. Set model to evaluation mode and move to device
    model = model.to(DEVICE)
    model.eval()
    logger.info("Model set to evaluation mode.")

except FileNotFoundError as e:
    logger.error(f"Error loading model: {e}")
    # Depending on deployment strategy, you might want the app to exit or run in a limited state.
    # For simplicity here, we'll let it proceed but endpoints might fail later.
    model = None # Indicate model loading failed
except Exception as e:
    logger.error(f"An unexpected error occurred during model loading: {e}")
    model = None


# --- Define Transforms (Aligned with full_pipe.py 'val' transform) ---
data_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Use IMG_SIZE = 224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Standard ImageNet normalization
])
logger.info("Data transforms defined.")


def analyze_video(video_path: str) -> Optional[dict]:
    """Analyze a video file and return results. Returns None if model not loaded."""
    if model is None:
        logger.error("Model is not loaded, cannot analyze video.")
        return None

    logger.info(f"Starting analysis for video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        logger.warning(f"Video {video_path} has no frames.")
        cap.release()
        return { # Return a result indicating no frames
             "file_name": os.path.basename(video_path),
             "total_frames": 0,
             "frames_analyzed": 0,
             "real_frames": 0,
             "fake_frames": 0,
             "verdict": "UNKNOWN (No frames)",
             "confidence": 0,
             "average_confidence": 0,
             "frame_predictions": [],
             "frame_confidences": []
         }

    # Select frame indices evenly spaced throughout the video
    num_frames_to_select = min(FRAMES_TO_ANALYZE, total_frames)
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames_to_select, dtype=int)
    logger.info(f"Total frames: {total_frames}. Analyzing {len(frame_indices)} frames.")

    predictions = []
    confidence_scores = []

    try:
        with torch.no_grad(): # Ensure no gradients are calculated
            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR (OpenCV default) to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert numpy array to PIL Image (required by transforms)
                    pil_img = Image.fromarray(frame_rgb)
                    # Apply transforms and add batch dimension
                    img_tensor = data_transform(pil_img).unsqueeze(0).to(DEVICE)
                    # Perform inference
                    output = model(img_tensor)
                    # Get probabilities using softmax
                    probs = torch.nn.functional.softmax(output, dim=1)
                    # Get the predicted class index (0=Real, 1=Fake)
                    pred = torch.argmax(output).item()
                    # Get the confidence score for the predicted class
                    confidence = probs[0][pred].item() * 100

                    predictions.append(pred)
                    confidence_scores.append(confidence)
                else:
                    logger.warning(f"Could not read frame index {frame_index} from {video_path}")

    except Exception as e:
        logger.error(f"Error during model inference for {video_path}: {e}")
        cap.release()
        return None # Indicate analysis failure
    finally:
        cap.release() # Ensure video file is released

    if not predictions:
        logger.warning(f"No frames were successfully processed for {video_path}")
        return { # Return a result indicating no frames processed
             "file_name": os.path.basename(video_path),
             "total_frames": total_frames,
             "frames_analyzed": 0,
             "real_frames": 0,
             "fake_frames": 0,
             "verdict": "UNKNOWN (Processing Error)",
             "confidence": 0,
             "average_confidence": 0,
             "frame_predictions": [],
             "frame_confidences": []
         }


    # Calculate overall results
    real_count = predictions.count(0)
    fake_count = predictions.count(1)
    total_frames_analyzed = len(predictions)
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

    # Determine overall verdict (0 = real, 1 = fake) based on majority vote
    overall_verdict_idx = 0 if real_count >= fake_count else 1 # Default to real if counts are equal
    overall_verdict_str = "REAL" if overall_verdict_idx == 0 else "FAKE"

    # Calculate confidence based on the proportion of frames matching the final verdict
    if total_frames_analyzed > 0:
         verdict_confidence = (real_count / total_frames_analyzed * 100) if overall_verdict_idx == 0 else (fake_count / total_frames_analyzed * 100)
    else:
         verdict_confidence = 0

    logger.info(f"Analysis complete for {video_path}. Verdict: {overall_verdict_str}, Confidence: {verdict_confidence:.2f}%")

    return {
        "file_name": os.path.basename(video_path),
        "total_frames": total_frames,
        "frames_analyzed": total_frames_analyzed,
        "real_frames": real_count,
        "fake_frames": fake_count,
        "verdict": overall_verdict_str,
        "confidence": round(verdict_confidence, 2),
        "average_confidence": round(avg_confidence, 2), # Average confidence of individual frame predictions
        "frame_predictions": predictions, # 0=Real, 1=Fake for each analyzed frame
        "frame_confidences": [round(c, 2) for c in confidence_scores] # Confidence for each frame prediction
    }

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main page"""
    logger.info("Serving index page.")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_uploaded_video(file: UploadFile = File(...)):
    """Handle video upload and analysis"""
    if model is None:
        logger.error("Model not loaded, cannot process request.")
        return JSONResponse(
            status_code=503, # Service Unavailable
            content={"success": False, "error": "Model is not available. Please check server logs."}
        )

    # Generate unique filename to avoid collisions
    file_ext = os.path.splitext(file.filename)[1]
    if file_ext.lower() not in ['.mp4', '.avi', '.mov', '.mkv']: # Basic video format check
         logger.warning(f"Unsupported file type uploaded: {file.filename}")
         return JSONResponse(
             status_code=400, # Bad Request
             content={"success": False, "error": "Unsupported file type. Please upload a video (MP4, AVI, MOV, MKV)."}
         )

    unique_id = str(uuid.uuid4())
    upload_path = os.path.join(UPLOAD_DIR, f"{unique_id}{file_ext}")
    logger.info(f"Receiving file: {file.filename}, saving to: {upload_path}")

    try:
        # Save the uploaded file securely
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved successfully: {upload_path}")

        # Analyze the video
        results = analyze_video(upload_path)

        if results is None:
             # Error occurred during analysis (already logged in analyze_video)
             return JSONResponse(
                 status_code=500,
                 content={"success": False, "error": "Video analysis failed. Check server logs."}
             )

        # Prepare successful response
        response_data = {
            "success": True,
            "results": results
        }
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.exception(f"An error occurred during file upload or analysis for {file.filename}: {e}") # Log traceback
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"An internal server error occurred: {e}"}
        )
    finally:
        # Clean up the uploaded file after analysis (optional)
        if os.path.exists(upload_path):
             try:
                 os.remove(upload_path)
                 logger.info(f"Cleaned up uploaded file: {upload_path}")
             except OSError as e:
                 logger.error(f"Error removing uploaded file {upload_path}: {e}")


# This endpoint remains a placeholder as result storage isn't implemented.
# If you need persistent results, you'd need a database or file-based storage.
@app.get("/results/{result_id}")
async def get_analysis_results(result_id: str):
    """Retrieve analysis results by ID (Placeholder)"""
    logger.warning(f"Received request for results ID {result_id}, but result storage is not implemented.")
    return JSONResponse(
        status_code=404, # Not Found
        content={"error": "Result retrieval by ID is not implemented in this version."}
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Luna Shield Deepfake Detector API...")
    # Check if model loaded correctly before starting server
    if model is None:
        logger.critical("Model failed to load. The API will start, but analysis endpoints will return errors.")
        # Decide if you want to prevent startup:
        # import sys
        # sys.exit("Exiting due to model loading failure.")
    uvicorn.run(app, host="0.0.0.0", port=8000)