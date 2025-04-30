import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from config.config import Config
from models.rediffinet import ReDiffiNet
from utils.visualization import visualize_results
import matplotlib.pyplot as plt

def load_image(image_path, target_size=None):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    if target_size:
        img = img.resize(target_size)
    
    # Convert to numpy array and normalize
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Add channel dimension and convert to tensor
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)
    
    return img_tensor

def run_inference(model, image_path, device, target_size=(256, 256)):
    """Run inference on a single image"""
    model.eval()
    
    # Load and preprocess image
    img = load_image(image_path, target_size=target_size)
    img = img.to(device)
    
    # Run inference with no gradients
    with torch.no_grad():
        # Get U-Net prediction
        unet_pred = model.unet(img)
        unet_pred_sigmoid = torch.sigmoid(unet_pred)
        
        # Refine prediction using diffusion model
        refined_pred = model._refine_predictions(img, unet_pred_sigmoid)
    
    return {
        "image": img.cpu(),
        "unet_pred": unet_pred_sigmoid.cpu(),
        "refined_pred": refined_pred.cpu()
    }

def save_segmentation_overlay(image, pred, save_path, colors=None):
    """Save segmentation overlay"""
    if colors is None:
        colors = [
            [1, 0, 0, 0.5],  # Red for class 0
            [0, 1, 0, 0.5],  # Green for class 1
            [0, 0, 1, 0.5],  # Blue for class 2
        ]
    
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy().squeeze()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Plot original image
    plt.imshow(image, cmap='gray')
    
    # Overlay segmentation masks
    for c in range(pred.shape[0]):
        mask = pred[c] > 0.5
        if mask.sum() > 0:  # Only plot if mask has content
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask, :] = colors[c]
            plt.imshow(colored_mask)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path)
    plt.close()

def main():
    config = Config()
    device = torch.device(config.device)
    
    # Load model
    model = ReDiffiNet(
        in_channels=config.unet_in_channels, 
        out_channels=config.num_classes,
        unet_base_filters=config.unet_base_filters,
        unet_depth=config.unet_depth,
        diffusion_steps=config.diffusion_steps,
        beta_schedule=config.beta_schedule,
        beta_start=config.beta_start,
        beta_end=config.beta_end
    ).to(device)
    
    # Load weights
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'weights', 'rediffinet_best.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
        return
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Example: Run inference on a single image
    image_path = "path/to/test_image.png"  # Replace with actual path
    results = run_inference(model, image_path, device)
    
    # Save results
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save U-Net prediction
    unet_save_path = os.path.join(results_dir, f"{image_name}_unet.png")
    save_segmentation_overlay(
        results["image"], 
        results["unet_pred"],
        unet_save_path
    )
    
    # Save refined prediction
    refined_save_path = os.path.join(results_dir, f"{image_name}_refined.png")
    save_segmentation_overlay(
        results["image"], 
        results["refined_pred"],
        refined_save_path
    )
    
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()