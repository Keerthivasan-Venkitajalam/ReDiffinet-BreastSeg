import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import UNet
from models.diffusion import DiffusionUNet, GaussianDiffusion

class ReDiffiNet(nn.Module):
    def __init__(
        self, 
        in_channels=1,
        out_channels=3,
        unet_base_filters=64,
        unet_depth=5,
        diffusion_steps=1000,
        beta_schedule='linear',
        beta_start=0.0001,
        beta_end=0.02
    ):
        super().__init__()
        
        # Baseline U-Net for initial segmentation
        self.unet = UNet(in_channels, out_channels, base_filters=unet_base_filters, depth=unet_depth)
        
        # Diffusion model for discrepancy prediction
        # Input is concatenation of: 
        # - U-Net prediction (out_channels)
        # - Original image (in_channels)
        # - Noised discrepancy mask (out_channels)
        self.diffusion_unet = DiffusionUNet(
            in_channels=in_channels + 2*out_channels,
            out_channels=out_channels,
            base_filters=unet_base_filters,
            depth=unet_depth
        )
        
        # Diffusion parameters
        self.diffusion = GaussianDiffusion(
            timesteps=diffusion_steps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end
        )
        
    def forward(self, x):
        # Get U-Net prediction
        unet_pred = self.unet(x)
        unet_pred_sigmoid = torch.sigmoid(unet_pred)
        
        # Just return U-Net prediction during inference
        if not self.training:
            return self._refine_predictions(x, unet_pred_sigmoid)
        
        return unet_pred
    
    def calculate_discrepancy(self, unet_pred, target):
        """
        Calculate absolute difference between U-Net prediction and ground truth
        """
        unet_sigmoid = torch.sigmoid(unet_pred)
        discrepancy = torch.abs(unet_sigmoid - target)
        return discrepancy
    
    def calculate_discrepancy_loss(self, x, unet_pred, target, t=None):
        """
        Calculate loss for diffusion model predicting discrepancy
        """
        # Calculate discrepancy
        discrepancy = self.calculate_discrepancy(unet_pred, target)
        
        # Sample random timestep if not provided
        if t is None:
            t = torch.randint(0, self.diffusion.timesteps, (x.shape[0],), device=x.device).long()
        
        # Forward diffusion on discrepancy
        noise = torch.randn_like(discrepancy)
        noisy_discrepancy = self.diffusion.q_sample(discrepancy, t, noise=noise)
        
        # Concat input for diffusion model: [UNet prediction, original image, noisy discrepancy]
        unet_sigmoid = torch.sigmoid(unet_pred)
        diffusion_input = torch.cat([unet_sigmoid, x, noisy_discrepancy], dim=1)
        
        # Run diffusion model to predict noise
        predicted_noise = self.diffusion_unet(diffusion_input, t)
        
        # Calculate loss
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    def sample_discrepancy(self, x, unet_pred, steps=None):
        """
        Sample discrepancy from diffusion model
        """
        if steps is None:
            steps = self.diffusion.timesteps
            
        # Start from random noise
        b, c, h, w = unet_pred.shape
        device = unet_pred.device
        
        discrepancy = torch.randn(b, c, h, w, device=device)
        
        unet_sigmoid = torch.sigmoid(unet_pred)
        
        # Iteratively denoise
        for t in reversed(range(0, steps)):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            # Concat input for diffusion model
            diffusion_input = torch.cat([unet_sigmoid, x, discrepancy], dim=1)
            
            # Predict noise
            predicted_noise = self.diffusion_unet(diffusion_input, t_batch)
            
            # Update sample
            alpha_t = self.diffusion.alphas[t]
            alpha_prev = self.diffusion.alphas_cumprod_prev[t]
            beta_t = self.diffusion.betas[t]
            
            if t > 0:
                noise = torch.randn_like(discrepancy)
            else:
                noise = torch.zeros_like(discrepancy)
                
            # Sample x_{t-1} from p(x_{t-1} | x_t, x_0)
            discrepancy = (
                1 / torch.sqrt(alpha_t) * (
                    discrepancy - 
                    (beta_t / torch.sqrt(1 - alpha_prev)) * predicted_noise
                ) + 
                torch.sqrt(beta_t) * noise
            )
            
        return discrepancy
    
    def _refine_predictions(self, x, unet_pred):
        """
        Refine U-Net predictions using diffusion model
        """
        # Sample discrepancy from diffusion model
        discrepancy = self.sample_discrepancy(x, unet_pred)
        
        # Apply discrepancy correction
        refined_pred = torch.abs(unet_pred - discrepancy)
        
        return refined_pred