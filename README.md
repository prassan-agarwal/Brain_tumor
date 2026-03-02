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

## Create a brand-new repo (no merge with this repo)

If you want a separate repository with the current files and **no link/merge history** back to this one:

```bash
bash scripts/create_standalone_repo.sh ../Brain_tumor_standalone
```

This will:

1. Copy project files to a new folder.
2. Remove any copied `.git` metadata.
3. Initialize a fresh Git repository there.
4. Create an initial commit (`Initial standalone import from Brain_tumor`).

You can then add a new remote in the standalone folder and push it as an independent repo:

```bash
cd ../Brain_tumor_standalone
git remote add origin <your-new-repo-url>
git push -u origin main
```

## Notes

This project is for educational/research use and not for clinical decision-making.
