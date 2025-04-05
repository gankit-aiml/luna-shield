#latest code
import os
import random
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

# Optimized Configuration
VIDEO_DIR = "dataset"
EXTRACTED_FRAMES_DIR = "extracted frames"
BATCH_SIZE = 64
NUM_EPOCHS = 10
IMG_SIZE = 224
LEARNING_RATE = 3e-4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
FRAMES_PER_VIDEO = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories
os.makedirs(EXTRACTED_FRAMES_DIR, exist_ok=True)
os.makedirs("./report_charts", exist_ok=True)

# Simplified Data Augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def extract_frames(video_dir, output_dir, frames_per_video=10):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for cls in ['real', 'fake']:
        class_path = os.path.join(video_dir, cls)
        videos = os.listdir(class_path)
        save_dir = os.path.join(output_dir, cls)
        os.makedirs(save_dir, exist_ok=True)

        for vid in tqdm(videos, desc=f"Extracting {cls} frames"):
            cap = cv2.VideoCapture(os.path.join(class_path, vid))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_ids = sorted(random.sample(range(frame_count), min(frames_per_video, frame_count)))

            for i, fid in enumerate(frame_ids):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                success, frame = cap.read()
                if success:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    filename = f"{os.path.splitext(vid)[0]}_frame{i}.jpg"
                    cv2.imwrite(os.path.join(save_dir, filename), frame)
            cap.release()

def split_dataset(base_dir, output_dir):
    for cls in ['real', 'fake']:
        class_path = os.path.join(base_dir, cls)
        files = os.listdir(class_path)
        random.shuffle(files)

        train_end = int(len(files) * TRAIN_RATIO)
        val_end = int(len(files) * (TRAIN_RATIO + VAL_RATIO))

        splits = {
            'train': files[:train_end],
            'val': files[train_end:val_end],
            'test': files[val_end:]
        }

        for split, split_files in splits.items():
            dest_dir = os.path.join(output_dir, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for f in split_files:
                shutil.copy(os.path.join(class_path, f), os.path.join(dest_dir, f))

# Optimized Model Setup
model = models.efficientnet_b0(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_ftrs, 2)
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

def train_model():
    best_acc = 0.0
    scaler = GradScaler()
    datasets_dict = {x: datasets.ImageFolder(os.path.join('./split_data', x), data_transforms[x])
                    for x in ['train', 'val']}

    dataloaders = {x: DataLoader(datasets_dict[x],
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True,
                                persistent_workers=True)
                  for x in ['train', 'val']}

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        # Training Phase
        model.train()
        train_loss, train_correct = 0.0, 0
        for inputs, labels in tqdm(dataloaders['train'], desc="Training"):
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_correct += torch.sum(preds == labels.data)

        epoch_loss = train_loss / len(datasets_dict['train'])
        epoch_acc = train_correct.double() / len(datasets_dict['train'])
        print(f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        # Validation Phase
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(dataloaders['val'], desc="Validating"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_correct += torch.sum(preds == labels.data)

        val_loss /= len(datasets_dict['val'])
        val_acc = val_correct.double() / len(datasets_dict['val'])
        scheduler.step(val_loss)

        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}\n")

        # Early Stopping
        if epoch > 2 and val_acc < best_acc * 0.95:
            print("Early stopping triggered")
            break

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")

def evaluate_model():
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    dataset = datasets.ImageFolder(os.path.join('./split_data', 'test'), data_transforms['val'])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('./report_charts/confusion_matrix.png')
    plt.close()

    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))

def generate_report(video_path):
    # Frame extraction and prediction
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, num=FRAMES_PER_VIDEO, dtype=int)
    predictions = []

    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    transform = data_transforms['val']

    with torch.no_grad():
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = transform(frame).unsqueeze(0).to(DEVICE)
                output = model(img)
                predictions.append(torch.argmax(output).item())

    cap.release()

    # Generate visualization
    plt.figure(figsize=(10,4))
    plt.bar(range(len(predictions)), predictions,
            color=['green' if p==0 else 'red' for p in predictions])
    plt.title('Frame-wise Predictions')
    plt.xlabel('Frame Index')
    plt.ylabel('Prediction (0=Real, 1=Fake)')
    plt.savefig('./report_charts/frame_predictions.png')
    plt.close()

    # Create PDF report
    doc = SimpleDocTemplate("Deepfake_Report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Deepfake Detection Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Confusion Matrix
    story.append(Paragraph("Model Performance:", styles['Heading2']))
    story.append(Image('./report_charts/confusion_matrix.png', width=400, height=300))
    story.append(Spacer(1, 12))

    # Predictions
    story.append(Paragraph("Video Analysis Results:", styles['Heading2']))
    story.append(Image('./report_charts/frame_predictions.png', width=500, height=200))
    story.append(Spacer(1, 12))

    # Statistics
    real_count = predictions.count(0)
    fake_count = predictions.count(1)
    total = len(predictions)
    data = [
        ["Metric", "Value"],
        ["Total Frames Analyzed", total],
        ["Real Frames Detected", f"{real_count} ({real_count/total:.1%})"],
        ["Fake Frames Detected", f"{fake_count} ({fake_count/total:.1%})"],
        ["Final Conclusion", "REAL CONTENT" if real_count > fake_count else "DEEPFAKE DETECTED"]
    ]

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(table)

    story.append(Spacer(1, 24))
    story.append(Paragraph(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Italic']))

    doc.build(story)
    print("Report saved as Deepfake_Report.pdf")

# Execution Pipeline
if __name__ == "__main__":
    # Step 1: Data Preparation
    print("Extracting frames from videos...")
    extract_frames(VIDEO_DIR, EXTRACTED_FRAMES_DIR)
    print("Splitting dataset...")
    split_dataset(EXTRACTED_FRAMES_DIR, './split_data')

    # Step 2: Model Training
    print("\nStarting model training...")
    train_model()

    # Step 3: Evaluation
    print("\nEvaluating model...")
    evaluate_model()

    # Step 4: Generate Report
    test_video = "notfake_vid.mp4"  # Replace with your video path
    print(f"\nAnalyzing {test_video}...")
    generate_report(test_video)