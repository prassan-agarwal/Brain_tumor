# Advanced Brain Tumor Diagnosis & Analysis

This project is an advanced, end-to-end Machine Learning pipeline for diagnosing brain tumors from MRI scans. It leverages an ensemble of Deep Learning models, Weakly-Supervised Localization for explainability, and Out-of-Distribution Anomaly Detection to validate inputs. The entire system is packaged into a local web application using Gradio.

## Features
- **Ensemble Classification (VGG16 + ResNet50):** Uses a soft-voting ensemble of two varied CNN architectures to accurately classify MRI images into four classes: *Glioma*, *Meningioma*, *Pituitary*, or *No Tumor*.
- **Out-of-Distribution Anomaly Detection:** An unsupervised Autoencoder strictly evaluates uploaded images against a mathematical Mean Squared Error (MSE) threshold. This guarantees the system automatically rejects non-MRI portraits, noise, or unrelated pictures before classification.
- **Explainable AI (Grad-CAM):** Generates and overlays a gradient-weighted class activation heatmap to visually highlight the focal regions of the brain that influenced the network's decision.
- **Weakly-Supervised Localization:** Uses contour thresholding on the Grad-CAM heatmaps to estimate tumor coordinates and percentage of brain size without requiring pixel-perfect ground truth masks during training.
- **Interactive Web UI:** A sleek, user-friendly Gradio interface for uploading images and observing the diagnosis, Grad-CAM visualization, and system analysis in real-time.

## Project Structure
- `Brain_Tumor.ipynb`: The core Jupyter Notebook containing the development trajectory: data exploration, data generation, autoencoder construction, model ensembling, Grad-CAM definitions, and model export pipelines. 
- `app.py`: The deployment application script that loads the trained weights and locally hosts the Gradio web interface.
- `requirements.txt`: The isolated dependencies required to execute the models and frontend.
- `MRI Images/`: (If available locally) The source dataset organized by tumor classes.

## Getting Started

### 1. Installation
Clone the repository and install the dependencies. It is recommended to use a virtual environment.
```bash
git clone https://github.com/your-username/brain-tumor-diagnosis
cd brain-tumor-diagnosis
pip install -r requirements.txt
```

### 2. Generate the Models (Required)
Due to GitHub's file size limits, the heavy `.keras` model weights are not included in the repository. You must generate them locally first.
1. Open `Brain_Tumor.ipynb` in Jupyter Notebook or VS Code.
2. Run the entire notebook (`Run All`). 
3. The execution will automatically save `vgg16_tumor_model.keras`, `resnet50_tumor_model.keras`, `autoencoder_tumor_model.keras`, and `anomaly_threshold.txt` to your directory.

### 3. Launch the Web Application
Once the models are generated, you can boot the UI.
```bash
python app.py
```
A local URL (typically `http://127.0.0.1:7860/`) will be provided in the terminal. Open it in any browser to use the analyzer!

## Technical Stack
- **Frameworks:** TensorFlow / Keras (Backend), Gradio (Frontend UI)
- **Image Processing:** OpenCV (cv2), Pillow (PIL)
- **Data Math:** NumPy
