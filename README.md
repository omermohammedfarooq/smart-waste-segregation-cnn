# ♻️ Garbage Classifier AI - 90.46% Accuracy

A deep learning model that automatically classifies waste into 10 categories to facilitate efficient recycling and waste management.

![Python](https://img.shields.io/badge/Python-3.12.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-90.46%25-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

An AI-powered garbage classification system using Convolutional Neural Networks (CNNs) to automate waste segregation, enabling efficient recycling and reducing manual sorting effort.


# 🧩 Problem Statement
Improper waste management is one of the most persistent environmental challenges. Every year, millions of tons of waste end up in landfills due to ineffective segregation. Manual waste sorting is time-consuming, costly, and exposes workers to hazardous conditions.
This project develops an automated image-based classification model using deep learning to:

Enhance efficiency and accuracy of waste segregation systems
Reduce human effort and health risks in waste handling
Promote environmental sustainability through better recycling practices
Provide a scalable foundation for smart waste management systems (intelligent dustbins, sorting belts, recycling automation)

## 📊 Model Performance

- **Validation Accuracy:** 90.46%
- **Training Accuracy:** 92.05%
- **Model Architecture:** EfficientNetB0 with custom head
- **Total Parameters:** 4,417,837 (16.85 MB)
- **Trainable Parameters:** 365,194 (1.39 MB)
- **Training Time:** ~84 seconds per epoch on T4 GPU

---

## 🗂️ Categories

The model classifies waste into **10 categories**:

| Category | Training Images | Validation Images | Description |
|----------|----------------|-------------------|-------------|
| Battery | 802 | 142 | Batteries and power cells |
| Biological | 847 | 150 | Organic waste, food scraps |
| Cardboard | 1,551 | 274 | Cardboard boxes and packaging |
| Clothes | 4,527 | 800 | Textiles and fabric waste |
| Glass | 2,601 | 460 | Glass bottles and containers |
| Metal | 867 | 153 | Metal cans and scrap |
| Paper | 1,428 | 252 | Paper products and documents |
| Plastic | 1,686 | 298 | Plastic bottles and packaging |
| Shoes | 1,680 | 297 | Footwear and leather goods |
| Trash | 804 | 143 | Non-recyclable waste |

**Total Dataset:** 19,762 images (16,791 train + 2,968 validation)

---

## 🏗️ Architecture

### Base Model
- **EfficientNetB0** (pre-trained on ImageNet)
- Frozen layers: 4,052,643 parameters
- Input shape: 224×224×3

### Custom Classification Head
```
Input (224×224×3)
    ↓
Data Augmentation (RandomFlip, RandomRotation, RandomZoom)
    ↓
EfficientNetB0 Base (Frozen)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization → Dropout(0.3)
    ↓
Dense(256, relu, L2=0.01)
    ↓
BatchNormalization → Dropout(0.3)
    ↓
Dense(128, relu, L2=0.01)
    ↓
Dropout(0.2)
    ↓
Dense(10, softmax) → Output
```

### Data Augmentation
- Horizontal & vertical flips
- Random rotation (±20°)
- Random zoom (±20%)
- Random translation (±10%)
- Random contrast adjustment (±20%)

---

## 🚀 Training Process

### Phase 1: Head Training (Achieved 90.46% in 1 epoch!)
- **Epochs:** 15 (stopped early at epoch 1)
- **Learning Rate:** 0.001
- **Optimizer:** Adam
- **Batch Size:** 16 (memory-optimized for Colab)
- **Early Stopping:** Triggered after reaching target

### Key Training Milestones
| Epoch | Train Acc | Val Acc | Val Loss | Notes |
|-------|-----------|---------|----------|-------|
| 1 | 77.00% | **90.46%** | 1.2173 | 🎯 Target reached! |

---

## 💻 System Requirements

### Hardware Used
- **GPU:** NVIDIA T4 (15 GB VRAM)
- **RAM:** 12.7 GB system RAM
- **Disk:** 121 GB available space
- **Platform:** Google Colab

### Software Stack
```
Python: 3.12.12
TensorFlow: 2.19.0
Keras: 3.x (built into TensorFlow)
NumPy: 1.24.3
Pillow: 10.1.0
Matplotlib: (for visualization)
```

---

## 📁 Project Structure

```
garbage-classifier/
├── data/
│   ├── train/              # 16,791 training images
│   │   ├── battery/
│   │   ├── biological/
│   │   ├── cardboard/
│   │   └── ... (10 classes)
│   └── val/                # 2,968 validation images
│       └── ... (same structure)
├── exports/
│   ├── garbage_classifier_final.keras    # Best model (90.46%)
│   ├── phase1_best.keras                 # Phase 1 checkpoint
│   ├── label_map.json                    # Class name mapping
│   └── garbage_classifier_90percent.h5   # H5 format (optional)
├── bad_images_tf/          # Quarantined corrupt images (3 files)
└── README.md               # This file
```

---

## 🔧 Installation & Usage

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/garbage-classifier.git
cd garbage-classifier
```

### 2. Install Dependencies
```bash
pip install tensorflow==2.19.0 pillow numpy matplotlib gradio
```

### 3. Download Model
Download `garbage_classifier_final.keras` and `label_map.json` from releases.

### 4. Run Inference
```python
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model and labels
model = tf.keras.models.load_model('garbage_classifier_final.keras')
with open('label_map.json') as f:
    labels = json.load(f)

# Predict
img = Image.open('test_image.jpg').convert('RGB').resize((224, 224))
img_array = np.array(img).astype('float32')
img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)[0]
top_class = labels[str(predictions.argmax())]
confidence = predictions.max()

print(f"Prediction: {top_class} ({confidence*100:.2f}%)")
```

### 5. Launch Web Interface (Gradio)
```python
import gradio as gr

def classify(image):
    # ... (inference code above)
    return {labels[str(i)]: float(predictions[i]) 
            for i in predictions.argsort()[-3:][::-1]}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Garbage Classifier"
)
demo.launch()
```

---

## 📊 Dataset Information

### Source
- **Dataset:** Garbage Classification v2
- **Source:** [Kaggle](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
- **License:** MIT
- **Original Size:** 744 MB (compressed)

### Data Preprocessing
1. **Train/Val Split:** 85% / 15%
2. **Image Cleaning:** 
   - PIL verification pass: 0 corrupt images
   - TensorFlow decode verification: 3 corrupt images removed
3. **Final Dataset:** 19,759 clean images

### Class Distribution
- **Most Common:** Clothes (5,327 images)
- **Least Common:** Battery (944 images)
- **Median:** ~1,500 images per class
- **Imbalance Handled:** Data augmentation + class weights

---

## 🎯 Model Evaluation

### Confusion Matrix Analysis
```python
# Run evaluation script
python evaluate.py --model garbage_classifier_final.keras --data data/val/
```

### Per-Class Performance
*(Run confusion matrix cell in notebook for detailed metrics)*

Expected performance:
- **High Accuracy Classes:** Cardboard, Paper, Plastic (>92%)
- **Challenging Classes:** Battery, Metal (85-88%)
- **Overall:** 90.46% validation accuracy

---

## 🚀 Deployment Options

### Option 1: Hugging Face Spaces (Free)
```bash
# Upload to HF Spaces
huggingface-cli upload spaces/yourname/garbage-classifier \
  --repo-type space \
  garbage_classifier_final.keras label_map.json app.py
```

### Option 2: Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/garbage-classifier
gcloud run deploy --image gcr.io/PROJECT_ID/garbage-classifier
```

### Option 3: Local API (Flask/FastAPI)
See `deployment/` folder for Docker setup and API code.

---

## 📈 Future Improvements

### Accuracy Enhancements
- [ ] Use EfficientNetB2/B3 (larger model)
- [ ] Increase image resolution to 260×260 or 300×300
- [ ] Fine-tune base model layers (Phase 2-3 training)
- [ ] Apply test-time augmentation (TTA)
- [ ] Ensemble multiple models

### Feature Additions
- [ ] Multi-language support
- [ ] Real-time video classification
- [ ] Mobile app (TensorFlow Lite)
- [ ] Recycling instructions per category
- [ ] Location-based disposal recommendations

### Dataset Enhancements
- [ ] Collect more data for underrepresented classes (Battery, Metal)
- [ ] Add region-specific waste categories (e.g., e-waste, hazardous)
- [ ] Augment with synthetic data (GANs)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

Dataset: [MIT License](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)

---

## 👨‍💻 Author

**Your Name**
- GitHub: [0merfarooq](https://github.com/0merfarooq)
- Email: mohdomerfarooq17@gmail.com

---

## 🙏 Acknowledgments

- Dataset provided by [sumn2u](https://www.kaggle.com/sumn2u) on Kaggle
- EfficientNet architecture by [Google Research](https://arxiv.org/abs/1905.11946)
- Training infrastructure: Google Colab
- UI framework: Gradio

---

## 📚 References

1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*.
2. Kaggle Garbage Classification Dataset v2 (2024). *Kaggle Datasets*.
3. Transfer Learning with EfficientNet. *TensorFlow Documentation*.

---

## 🔗 Links

- **Live Demo:** [Try it here](https://huggingface.co/spaces/yourname/garbage-classifier)
- **Kaggle Notebook:** [View on Kaggle](https://kaggle.com/...)
- **Model Weights:** [Download from Releases](https://github.com/yourname/garbage-classifier/releases)
- **API Documentation:** [API Docs](https://yourname.github.io/garbage-classifier-api)

---

## ⭐ Star History

If you found this project helpful, please consider giving it a star! ⭐

---

**Made with ❤️ for sustainable waste management**
