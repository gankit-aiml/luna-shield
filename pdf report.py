# generate_video_report.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime
import argparse # For command-line arguments
from PIL import Image as PILImage # To use transforms with cv2 frames

# --- Configuration ---
MODEL_PATH = 'best_model.pth'   # Path to your trained model weights
REPORT_CHARTS_DIR = './report_charts' # Directory to save charts for the report
FRAMES_PER_VIDEO = 15 # Number of frames to sample from the video
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Ensure output directory exists ---
os.makedirs(REPORT_CHARTS_DIR, exist_ok=True)

# --- Data Transformations (use validation transforms) ---
data_transform = transforms.Compose([
    # transforms.ToPILImage(), # Convert numpy array (from cv2) to PIL Image
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Load Model ---
print(f"Loading model architecture...")
# Use the same architecture as during training
model = models.efficientnet_b0(weights=None) # Load architecture, not pretrained weights here
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_ftrs, 2) # Assuming 2 classes: real (0), fake (1)
)

print(f"Loading trained weights from {MODEL_PATH}...")
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except FileNotFoundError:
     print(f"Error: Model weights file not found at {MODEL_PATH}")
     exit()
except Exception as e:
    print(f"Error loading model weights: {e}")
    print("Ensure the model architecture defined here matches the one used for training.")
    exit()

model = model.to(DEVICE)
model.eval() # Set model to evaluation mode

# --- Function to Analyze Video ---
def analyze_video(video_path, num_frames_to_sample):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video {os.path.basename(video_path)} has {total_frames} frames.")

    if total_frames < 1:
         print(f"Error: Video {video_path} seems to have no frames.")
         cap.release()
         return None, None

    # Ensure num_frames_to_sample is not more than total_frames
    num_frames_to_sample = min(num_frames_to_sample, total_frames)

    # Sample frame indices uniformly
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames_to_sample, dtype=int)
    predictions = []
    processed_frame_indices = [] # Keep track of which frames were actually processed

    print(f"Analyzing {num_frames_to_sample} frames from the video...")
    with torch.no_grad():
        for idx in tqdm(frame_indices, desc="Processing frames"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = PILImage.fromarray(frame_rgb) # <--- THIS IS THE CRITICAL FIX
                    img_tensor = data_transform(pil_img).unsqueeze(0).to(DEVICE) # Apply transform to PIL image
                    # Get prediction
                    output = model(img_tensor)
                    pred = torch.argmax(output, dim=1).item() # 0 for real, 1 for fake
                    predictions.append(pred)
                    processed_frame_indices.append(idx)
                except Exception as e:
                    print(f"Warning: Could not process frame {idx}. Error: {e}")
            else:
                # This might happen if linspace generates an index slightly out of bounds
                # or the video has issues near the end.
                print(f"Warning: Could not read frame at index {idx}.")


    cap.release()

    if not predictions:
        print("Error: No frames could be processed from the video.")
        return None, None

    return predictions, processed_frame_indices


# --- Function to Generate PDF Report ---
def generate_pdf_report(video_path, predictions, frame_indices):
    if predictions is None:
        print("Skipping report generation due to analysis errors.")
        return

    video_filename = os.path.basename(video_path)
    report_filename = f"{os.path.splitext(video_filename)[0]}_Analysis_Report.pdf"
    doc = SimpleDocTemplate(report_filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Deepfake Analysis Report: {video_filename}", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Video Source: {os.path.abspath(video_path)}", styles['Normal']))
    story.append(Spacer(1, 12))

    # --- Frame Prediction Visualization ---
    if predictions:
        print("Generating frame prediction visualization...")
        plt.figure(figsize=(12, 5))
        colors_bar = ['green' if p == 0 else 'red' for p in predictions]
        # Use actual frame indices for x-axis labels if available and match length
        if frame_indices and len(frame_indices) == len(predictions):
             x_labels = [str(i) for i in frame_indices]
             plt.bar(x_labels, [p + 0.1 for p in predictions], color=colors_bar) # Offset slightly for visibility if needed
             plt.xlabel('Approx. Frame Index in Video')
        else:
             plt.bar(range(len(predictions)), [p + 0.1 for p in predictions], color=colors_bar)
             plt.xlabel('Sampled Frame Sequence Number')

        plt.ylabel('Prediction (0=Real, 1=Fake)')
        plt.yticks([0.1, 1.1], ['Real', 'Fake']) # Adjust ticks to match offset bars
        plt.title('Frame-wise Predictions')
        plt.tight_layout()
        chart_path = os.path.join(REPORT_CHARTS_DIR, f"{os.path.splitext(video_filename)[0]}_frame_predictions.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"Frame prediction chart saved to {chart_path}")

        story.append(Paragraph("Frame Analysis Visualization:", styles['Heading2']))
        try:
            story.append(Image(chart_path, width=500, height=200)) # Adjust size as needed
        except Exception as e:
             print(f"Error adding chart image to PDF: {e}. Check file permissions and path.")
             story.append(Paragraph(f"(Could not load chart image: {chart_path})", styles['Italic']))
        story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("Frame Analysis Visualization: No frames processed.", styles['Heading2']))
        story.append(Spacer(1,12))


    # --- Statistics Table ---
    story.append(Paragraph("Analysis Statistics:", styles['Heading2']))
    if predictions:
        real_count = predictions.count(0)
        fake_count = predictions.count(1)
        total_analyzed = len(predictions)
        real_percent = (real_count / total_analyzed) * 100 if total_analyzed > 0 else 0
        fake_percent = (fake_count / total_analyzed) * 100 if total_analyzed > 0 else 0
        # Simple conclusion based on majority vote
        final_conclusion = "PREDICTION: LIKELY REAL" if real_count >= fake_count else "PREDICTION: LIKELY FAKE"
        confidence = max(real_percent, fake_percent)

        data = [
            ["Metric", "Value"],
            ["Total Frames Sampled & Analyzed", total_analyzed],
            ["Frames Predicted as Real", f"{real_count} ({real_percent:.1f}%)"],
            ["Frames Predicted as Fake", f"{fake_count} ({fake_percent:.1f}%)"],
            ["Overall Assessment", f"{final_conclusion} ({confidence:.1f}% confidence)"]
        ]

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ]))
        story.append(table)
    else:
         story.append(Paragraph("No statistics available as no frames were processed.", styles['Normal']))


    story.append(Spacer(1, 24))
    story.append(Paragraph(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Italic']))
    story.append(Paragraph(f"Model used: {MODEL_PATH}", styles['Italic']))

    # --- Build the PDF ---
    try:
        doc.build(story)
        print(f"\nReport successfully saved as {report_filename}")
    except Exception as e:
        print(f"\nError building PDF report: {e}")
        print("Please ensure you have write permissions in the current directory.")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a video file for deepfakes and generate a report.")
    parser.add_argument("video_path", help="Path to the video file to analyze.")
    parser.add_argument("-f", "--frames", type=int, default=FRAMES_PER_VIDEO,
                        help=f"Number of frames to sample from the video (default: {FRAMES_PER_VIDEO}).")

    args = parser.parse_args()

    print(f"\nStarting analysis for video: {args.video_path}")
    predictions, frame_indices = analyze_video(args.video_path, args.frames)

    if predictions is not None:
        generate_pdf_report(args.video_path, predictions, frame_indices)
    else:
        print("Video analysis failed. Report cannot be generated.")