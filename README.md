# DEPTHLAB – Edge-Preserving Depth Map Upsampling and Refinement

## Project Overview

DEPTHLAB is a depth map reconstruction and refinement system that improves low-resolution depth maps using classical computer vision techniques, deep learning, and a hybrid approach.

The system uses an RGB image as guidance to preserve object boundaries and refine depth discontinuities while reconstructing depth maps at different upsampling factors.

## Objectives

- Reconstruct high-resolution depth maps from low-resolution inputs.
- Compare classical depth upsampling techniques.
- Develop a CNN-based depth refinement model.
- Combine classical and deep learning methods using a hybrid approach.
- Evaluate reconstruction quality, runtime, and memory usage.

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Flask
- HTML, CSS
- Matplotlib
- Pandas
- Scikit-image

## Implemented Methods

### Classical Methods
- Nearest Neighbour
- Bilinear Interpolation
- Bicubic Interpolation
- Bilateral Filtering
- Joint Bilateral Filtering
- Guided Filtering

### Deep Learning
A residual CNN-based depth refinement model uses RGB guidance and an upsampled depth map to predict depth corrections.

### Hybrid Method
The hybrid method combines CNN predictions and Joint Bilateral Filtering:

`D_hybrid = alpha × D_CNN + (1 - alpha) × D_JBF`

## Upsampling Factors

The system supports:
- 2× upsampling
- 4× upsampling
- 8× upsampling

## Dataset

The project uses a processed subset of the NYU Depth V2 dataset.

The processed dataset contains 420 samples:
- Training: 300 samples
- Validation: 60 samples
- Testing: 60 samples

The original and processed dataset files are not included in this repository because of their size.

## Evaluation Metrics

The reconstruction methods are evaluated using:

- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Runtime analysis
- Memory usage analysis

Evaluation results, comparison graphs, and visual outputs are available in the `outputs/` directory.

## Project Structure

```text
P4_Depth_Reconstruction/
├── classical/
├── dataset/
├── evaluation/
├── models/
├── outputs/
├── training/
├── webapp/
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DEPTHLAB-Depth-Reconstruction.git
cd DEPTHLAB-Depth-Reconstruction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Flask application

```bash
python webapp/app.py
```

Open the local URL displayed in the terminal, usually:

`http://127.0.0.1:5000/`

## Notes

- The trained model checkpoint is available in `models/`.
- The original NYU Depth V2 dataset is not included.
- Training scripts require the processed dataset.
- The Flask application provides the depth reconstruction interface.

## Team Project

DEPTHLAB is an academic project focused on edge-preserving depth map reconstruction using classical computer vision, deep learning, and hybrid techniques.