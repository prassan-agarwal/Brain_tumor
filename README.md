# Brain Tumor Detection using VGG16

This repository contains a Jupyter notebook (`Brain_Tumor.ipynb`) for classifying brain tumor MRI images with transfer learning (VGG16 + TensorFlow/Keras).

## Updated runtime (Python + packages)

The project has been refreshed for modern Python environments:

- **Python 3.12**
- **TensorFlow 2.17+** (with `tf.keras`)
- **NumPy 1.26+**
- **Pandas 2.2+**
- **scikit-learn 1.5+**
- **Matplotlib 3.9+**
- **Seaborn 0.13+**
- **Pillow 10+**

A `requirements.txt` file is included with compatible minimum versions.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Place the dataset in this structure (relative to the repository root):

```text
MRI Images/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

By default, the notebook reads from:

- `./MRI Images/Training`
- `./MRI Images/Testing`

If your dataset is elsewhere, update `dataset_root` in the notebook.

## Notes

This project is for educational/research use and not for clinical decision-making.
