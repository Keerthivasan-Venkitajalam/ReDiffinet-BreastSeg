# ReDiffinet-BreastSeg
## ReDiffiNet for Breast Tumor Segmentation

NOTEBOOK LINK: https://www.kaggle.com/code/keerthivasansv/multi-modal-medical-imaging

ReDiffiNet is a modular framework that combines a baseline U-Net and Denoising Diffusion Probabilistic Models (DDPM) to refine breast tumor segmentation results. This project is designed to address challenges in segmenting breast tumor regions, such as masses, calcifications, and architectural distortions, with a focus on improving boundary accuracy.

---

## Project Structure

The project directory is organized as follows:

```
ReDiffinet-BreastSeg/
├── config/
│   └── config.py              # Configuration settings (e.g., training, model parameters)
├── data/
│   ├── dataset.py             # Data loading utilities
│   └── transforms.py          # Data preprocessing and augmentation
├── models/
│   ├── unet.py                # Baseline U-Net implementation
│   ├── diffusion.py           # Diffusion model implementation
│   └── rediffinet.py          # Combined ReDiffiNet model
├── utils/
│   ├── metrics.py             # Evaluation metrics (e.g., Dice score, HD95)
│   └── visualization.py       # Visualization utilities
├── train.py                   # Training script for U-Net and ReDiffiNet
└── inference.py               # Inference script for segmentation

# Data directory (needs to be created manually)
├── dataset/
│   ├── train/
│   │   ├── images/            # Training images (e.g., mammograms, MRI slices)
│   │   └── masks/             # Training masks (ground truth)
│   ├── val/
│   │   ├── images/            # Validation images
│   │   └── masks/             # Validation masks
│   └── test/
│       ├── images/            # Test images for inference
│       └── masks/             # Optional, ground truth masks for evaluation

# These directories will be created automatically during execution
├── weights/                   # Saved model weights
├── visualizations/            # Visualizations of predictions and discrepancies
└── results/                   # Final segmentation results for test images
```

---

## Features

- **Baseline U-Net**:
  - A standard U-Net architecture for initial breast tumor segmentation.
  
- **Denoising Diffusion Probabilistic Model (DDPM)**:
  - Refines U-Net predictions by modeling the discrepancy between predictions and ground truth.
  
- **ReDiffiNet Framework**:
  - Combines U-Net and DDPM to improve overall segmentation accuracy, particularly at boundary regions.

- **Metrics**:
  - Supports Dice similarity coefficient and 95% Hausdorff Distance (HD95) for evaluation.

- **Visualization**:
  - Generates overlays of segmentation results and discrepancies for better interpretability.

---

## Installation

### Prerequisites

Ensure you have Python 3.8 or higher installed. Install the required dependencies:

```bash
pip install torch torchvision matplotlib numpy scipy pillow monai tqdm
```

---

## Usage

### 1. Prepare Dataset

Organize your data in the following directory structure:

```
dataset/
├── train/
│   ├── images/  # Training images
│   └── masks/   # Training masks (ground truth)
├── val/
│   ├── images/  # Validation images
│   └── masks/   # Validation masks
└── test/
    ├── images/  # Test images for inference
    └── masks/   # Optional, ground truth masks for evaluation
```

Ensure each image and its corresponding mask have the same filename.

---

### 2. Configure Settings

Edit `config/config.py` to update parameters such as:

- `data_path`: Path to your dataset directory.
- `image_size`: Image dimensions (e.g., `(256, 256)`).
- `device`: Set to `"cuda"` for GPU or `"cpu"` for CPU.

---

### 3. Train the Model

Run the training script to train both the baseline U-Net and ReDiffiNet models:

```bash
python train.py
```

This will:
- Train a U-Net as the baseline model.
- Train ReDiffiNet to refine U-Net predictions using discrepancy modeling.
- Save the trained models in the `weights/` directory.
- Save visualization samples in the `visualizations/` directory.

---

### 4. Run Inference

To run inference on test images, use the following:

```bash
python inference.py
```

This will:
- Load the trained ReDiffiNet model from `weights/`.
- Perform segmentation on images in `dataset/test/images/`.
- Save results to the `results/` directory.

Results include:
- `[image_name]_unet.png`: Prediction from the baseline U-Net.
- `[image_name]_refined.png`: Refined prediction from ReDiffiNet.

---

## Example

After running inference, check the `results/` directory for visualizations of your test images. For example:

- **Input Image**: The original test image.
- **U-Net Prediction**: Initial segmentation from the baseline U-Net.
- **Refined Prediction**: Improved segmentation after applying ReDiffiNet.

---

## Evaluation Metrics

The framework supports the following metrics:

- **Dice Similarity Coefficient**:
  - Measures region-based segmentation accuracy.
  
- **95% Hausdorff Distance (HD95)**:
  - Measures boundary accuracy.

---

## Visualizations

Predictions and discrepancies are visualized as overlays:

- **Red**: Mass regions
- **Green**: Calcifications
- **Blue**: Architectural distortions

Visualizations are saved in the `visualizations/` directory during training.

---

## Troubleshooting

- **Memory Issues**: Reduce `batch_size` in `config.py` if you encounter GPU memory issues.
- **Inference Speed**: Reduce `diffusion_steps` in `config.py` for faster inference (at the potential cost of accuracy).
- **Missing Data**: Ensure all images and masks are properly aligned and named in the dataset directory.

---

## Acknowledgements

This implementation is inspired by the paper *"Re-DiffiNet: Modeling discrepancies in tumor segmentation using diffusion models"* and adapted for breast tumor segmentation tasks.

For any questions or issues, feel free to raise an issue in this repository.
