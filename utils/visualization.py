"""
Visualization utilities for ReDiffiNet breast segmentation results.

### DIFFUSION MODELS MATHEMATICAL FOUNDATIONS ###

# FORWARD DIFFUSION PROCESS:
# The forward process gradually adds noise to an image according to:
#   x_t = √(1-β_t) * x_{t-1} + √(β_t) * ε
# where:
#   - x_t is the image at time step t
#   - β_t is the noise schedule parameter
#   - ε ~ N(0,1) is random Gaussian noise

# DIRECT SAMPLING FORMULA:
# We can directly compute any noisy version from the original image:
#   x_t = √(α̅_t) * x_0 + √(1-α̅_t) * ε
# where:
#   - α̅_t = ∏_{i=1}^t (1-β_i) is the cumulative product

# REVERSE PROCESS:
# The reverse process aims to learn the conditional probability:
#   p(x_{t-1}|x_t)
# which is modeled as a Gaussian with learnable parameters

# OPTIMIZATION OBJECTIVE:
# We maximize a variational lower bound on the log-likelihood:
#   log p(x_0) ≥ E_q[log p(x_0|x_1) - KL(q(x_T|x_0)||p(x_T)) 
#              - ∑KL(q(x_{t-1}|x_t,x_0)||p(x_{t-1}|x_t))]

# SIMPLIFIED TRAINING OBJECTIVE:
# The final simplified objective is:
#   L = ||ε - ε_θ(x_t, t)||²
# where the model predicts the noise added at time step t

### RE-DIFFINET ARCHITECTURE ###

# DISCREPANCY DEFINITION:
#   Δx₀ = abs(U(I) - x₀)
# where:
#   - U(I) is the baseline U-Net prediction
#   - x₀ is the ground truth

# DIFFUSION MODEL OPERATION:
#   Δx̂₀ = DU(cat(U(I),I,xₜ),t,Îf)
# where:
#   - DU is the denoising U-Net
#   - cat(U(I),I,xₜ) concatenates U-Net prediction, original image, and noise
#   - t is the time embedding
#   - Îf represents multi-scale features

# FINAL PREDICTION FORMATION:
#   x̂₀ = abs(U(I) - Δx̂₀)
# This corrects the initial U-Net prediction using the predicted discrepancy
"""

import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_results(image, gt_mask, pred_mask, discrepancy=None, save_path=None):
    """
    Visualize results
    
    Args:
        image: Input image (C, H, W)
        gt_mask: Ground truth mask (C, H, W)
        pred_mask: Predicted mask (C, H, W)
        discrepancy: Discrepancy mask (C, H, W) if available
        save_path: Path to save the figure
    """
    # Convert to numpy arrays
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if discrepancy is not None and isinstance(discrepancy, torch.Tensor):
        discrepancy = discrepancy.detach().cpu().numpy()
    
    # If input has more than one channel, take the first one for visualization
    if image.shape[0] > 1:
        image = image[0]
    else:
        image = image.squeeze(0)
    
    # Number of tumor classes
    num_classes = gt_mask.shape[0]
    
    # Define colormap for overlays
    colors = [
        [1, 0, 0, 0.5],  # Red for class 0 (mass)
        [0, 1, 0, 0.5],  # Green for class 1 (calcification)
        [0, 0, 1, 0.5],  # Blue for class 2 (architectural distortion)
    ]
    
    if discrepancy is None:
        # Plot without discrepancy
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Input image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(image, cmap='gray')
        for c in range(num_classes):
            mask = gt_mask[c]
            if mask.sum() > 0:  # Only plot if mask has content
                colored_mask = np.zeros((*mask.shape, 4))
                colored_mask[mask > 0, :] = colors[c]
                axes[1].imshow(colored_mask)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        # Predicted mask
        axes[2].imshow(image, cmap='gray')
        for c in range(num_classes):
            mask = pred_mask[c]
            if mask.sum() > 0:  # Only plot if mask has content
                colored_mask = np.zeros((*mask.shape, 4))
                colored_mask[mask > 0, :] = colors[c]
                axes[2].imshow(colored_mask)
        axes[2].set_title("Prediction")
        axes[2].axis('off')
    
    else:
        # Plot with discrepancy
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Input image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(image, cmap='gray')
        for c in range(num_classes):
            mask = gt_mask[c]
            if mask.sum() > 0:  # Only plot if mask has content
                colored_mask = np.zeros((*mask.shape, 4))
                colored_mask[mask > 0, :] = colors[c]
                axes[1].imshow(colored_mask)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        # Predicted mask
        axes[2].imshow(image, cmap='gray')
        for c in range(num_classes):
            mask = pred_mask[c]
            if mask.sum() > 0:  # Only plot if mask has content
                colored_mask = np.zeros((*mask.shape, 4))
                colored_mask[mask > 0, :] = colors[c]
                axes[2].imshow(colored_mask)
        axes[2].set_title("Prediction")
        axes[2].axis('off')
        
        # Discrepancy
        axes[3].imshow(image, cmap='gray')
        for c in range(num_classes):
            mask = discrepancy[c]
            if mask.sum() > 0:  # Only plot if mask has content
                colored_mask = np.zeros((*mask.shape, 4))
                colored_mask[mask > 0, :] = colors[c]
                axes[3].imshow(colored_mask)
        axes[3].set_title("Discrepancy")
        axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()