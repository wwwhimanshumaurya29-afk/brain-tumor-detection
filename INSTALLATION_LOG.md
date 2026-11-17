
# Brain Tumor Detection — Installation Log

## 1. Project Information

- Project Name: Brain Tumor Detection
- Project Type: Machine Learning / Deep Learning Research
- Development Environment: Visual Studio Code
- Operating System: Windows 11
- Project Status: Initial Setup and Research

## 2. Software Inventory

| Software | Installed Version | Status |
|---|---|---|
| Python | 3.14.3 | Installed |
| pip (Global) | 26.1.2 | Installed |
| pip (Virtual Environment) | 26.2.2 | Installed |
| Git | 2.53.0.windows.1 | Installed |
| Node.js | 24.13.1 | Installed |
| npm | 11.8.0 | Installed |
| PyTorch | 2.15.0+cpu | Installed |
| torchvision | 0.29.2+cpu | Installed |

## 3. Python Dependencies

| Package | Version | Purpose |
|---|---|---|
| NumPy | 2.5.4 | Numerical operations |
| Pandas | 3.1.0 | Dataset handling |
| Matplotlib | 3.11.2 | Visualization |
| Scikit-learn | 1.9.1 | Machine learning |
| Pillow | 13.0.0 | Image processing |
| SciPy | 1.18.1 | Scientific computing |
| PyTorch | 2.15.0+cpu | Deep learning |
| torchvision | 0.29.2+cpu | Image transformations and models |

Note: Versions should be checked again before final release.

## 4. Python Environment

- Virtual environment folder: .venv
- Environment type: Python venv
- Activation command (PowerShell):

  .\.venv\Scripts\Activate.ps1

- Dependency file: requirements.txt

## 5. Hardware Information

- Laptop: HP 15s series
- Processor: Intel Core i5, 13th Generation
- RAM: 16 GB
- Graphics: Intel UHD integrated graphics
- PyTorch configuration: CPU-only
- CUDA availability: False

## 6. Project Directory Structure
Brain-Tumor-Detection/
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── research/
│   └── literature_review.md
│
├── notebooks/
│
├── src/
│   ├── config.py
│   ├── inspect_dataset.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── train.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── visualization.py
│
├── tests/
├── .venv/
├── .gitignore
├── README.md
├── INSTALLATION_LOG.md
└── requirements.txt

## 7. Dataset Status

- Dataset: Figshare Brain Tumor Dataset
- Author: Jun Cheng
- DOI: 10.6084/m9.figshare.1512427
- Download status: In progress
- Extraction status: Not started
- Dataset inspection: Pending
- Dataset licence and usage conditions: To be checked

## 8. Research Status

- Literature review: Initial draft prepared
- Paper 1: Bibliographic verification pending
- Paper 2: Reviewed
- Paper 3: Reviewed
- Research gap: Candidate, not finalized
- Dataset selection: Preliminary
- Model selection: Pending
- Experiments: Not started

## 9. GitHub Safety

- .venv/ must not be uploaded.
- Raw datasets must not be uploaded.
- Model weights must not be uploaded without checking size and licence.
- API keys and passwords must never be committed.
- requirements.txt and documentation should be maintained.

## 10. Future Installation and Experiment Notes

Record all future package installations, version changes,
commands, configuration changes and experiment dependencies here.
