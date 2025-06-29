
# 🛡️ LunaShield: AI-Powered Deepfake Detection Framework

**LunaShield** is a comprehensive deepfake detection framework that leverages cutting-edge computer vision and deep learning to analyze, classify, and report on the authenticity of video content.

It is built for **end-to-end forensic analysis** of deepfake videos—right from frame extraction, model training, evaluation, and finally, PDF reporting with visualizations. Ideal for researchers, journalists, digital forensics teams, or developers who want a production-ready solution.

---

## 📌 Key Features

- 🎞️ **Video Frame Extraction** — Captures keyframes from real/fake videos
- 🧠 **Deep Learning Engine** — EfficientNet-B0 architecture pretrained on ImageNet
- 🧪 **Advanced Data Splitting** — Stratified train/val/test with augmentation
- ⚡ **Mixed Precision Training** — With `torch.cuda.amp` for optimized GPU usage
- 📉 **Performance Monitoring** — Real-time loss/accuracy + confusion matrix plots
- 📝 **Automated Report Generation** — Clean PDF with predictions + decision summary
- 🧰 **Modular Codebase** — Well-separated extraction, training, evaluation, reporting

---

## 📂 Directory Structure

```
├── dataset/                   # Original input videos (real/fake)
├── extracted frames/         # Extracted image frames
├── split_data/               # Organized train/val/test sets
├── report_charts/            # Plots: confusion matrix, bar graphs
├── Deepfake_Report.pdf       # Final analysis report
└── full_pipe.py              # Main execution pipeline
```

---

## 🧠 Model Architecture

- **Backbone:** EfficientNet-B0 (from torchvision)
- **Custom Head:** Dropout + Linear (for binary classification)
- **Training Epochs:** 10 (with early stopping)
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (`lr=3e-4`)
- **Scheduler:** ReduceLROnPlateau (auto adjusts LR)
- **Precision:** Mixed (via AMP and GradScaler)
- **Input Size:** 224x224 RGB
- **Classes:** `Real`, `Fake`

---

## 🧪 Workflow Overview

### 1. 🔍 Video Preprocessing

```python
extract_frames(VIDEO_DIR, EXTRACTED_FRAMES_DIR)
```

### 2. 🔀 Dataset Splitting

```python
split_dataset(EXTRACTED_FRAMES_DIR, './split_data')
```

### 3. 📚 Training

```python
train_model()
```

### 4. 📊 Evaluation

```python
evaluate_model()
```

### 5. 📄 PDF Report Generation

```python
generate_report("your_video.mp4")
```

---

## 🧱 Dependencies

Install the following Python packages:

```bash
pip install torch torchvision opencv-python matplotlib seaborn scikit-learn reportlab tqdm
```

Or use this `requirements.txt`:

```
torch
torchvision
opencv-python
matplotlib
seaborn
scikit-learn
reportlab
tqdm
```

---

## 📈 Example Report Summary

| Metric                  | Value                  |
|-------------------------|------------------------|
| Total Frames Analyzed   | 10                     |
| Real Frames Detected    | 6 (60%)                |
| Fake Frames Detected    | 4 (40%)                |
| Final Conclusion        | REAL CONTENT           |

---

## 🛠️ Customization Guide

| Task                     | How to do it                                               |
|--------------------------|------------------------------------------------------------|
| Change number of frames  | Edit `FRAMES_PER_VIDEO` in `full_pipe.py`                 |
| Use your own videos      | Replace files in `dataset/real/` and `dataset/fake/`      |
| Change model backbone    | Swap EfficientNet with ResNet/VisionTransformer/etc.       |
| Tune hyperparameters     | Adjust `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, etc.  |
| Improve augmentations    | Edit `data_transforms` dictionary                          |
| Change report design     | Modify the `generate_report` function using ReportLab     |

---

## 🔍 Technical Highlights

- **Optimized for Speed**
- **Modular & Reusable**
- **Scalable**
- **Production-Ready**

---

## 📌 Roadmap / TODOs

- [ ] Add Streamlit-based UI
- [ ] Real-time webcam detection
- [ ] LSTM/CNN hybrid for spatio-temporal detection
- [ ] Audio-video multimodal support
- [ ] FastAPI deployment

---

## 🤝 Contributing

Contributions are welcome!

---

## 📃 License

MIT License

---


## ⭐️ If you like this project...

Please star ⭐️ the repo and share!
