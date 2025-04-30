"""
Implementation of Diffusion Models for ReDiffiNet breast segmentation.

### Mathematical Summary of DDPM (Denoising Diffusion Probabilistic Models) ###

#### Key Concepts:

1. Forward Process (Diffusion):
   - The forward process progressively adds Gaussian noise to an image over a series of time steps t, 
     transforming it into pure random noise by the final step.
   - The transition from x_{t-1} to x_t is governed by a Markov process:
     
     q(x_t | x_{t-1}) = N(x_t; √(α_t) x_{t-1}, β_t I)
     
     where:
     - α_t controls how much of the original structure remains
     - β_t controls the variance of the added noise

   - After T steps, the resulting distribution q(x_T) approximates a standard normal distribution:
     
     q(x_T) ≈ N(0, I)

2. Reverse Process (Denoising):
   - The reverse process learns to iteratively denoise x_t to recover the original image x_0.
   - The reverse transition is modeled as:
     
     p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
     
     where:
     - μ_θ is the predicted mean of the denoising distribution
     - Σ_θ is typically fixed to simplify training

3. Reparameterization:
   - To efficiently compute x_t at any time step t, the forward process can be reformulated as:
     
     x_t = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε
     
     where:
     - ᾱ_t = ∏_{s=1}^t α_s is the cumulative product of α_t
     - ε ~ N(0, I) is standard Gaussian noise

#### Training Objective:
- The training objective is to match the predicted noise ε_θ(x_t, t) with the true noise ε
  added during the forward process.
- The simplified loss function becomes:
  
  L = E_{t,x_0,ε} [||ε - ε_θ(x_t, t)||²]
  
  where:
  - t is sampled uniformly across all time steps
  - x_t is generated using the forward process

#### Practical Considerations:
1. Noise Schedule:
   - Variance β_t is scheduled to increase linearly over time.
   - Early steps add small noise (preserving structure), while later steps add larger noise.

2. One-Step Transition:
   - Instead of iteratively applying the forward process for t steps,
     x_t can be computed directly using the reparameterization formula.

3. Reverse Sampling:
   - To generate data, start with random Gaussian noise x_T ~ N(0, I) and 
     iteratively apply the learned reverse transitions p_θ(x_{t-1} | x_t).

#### Connections to Variational Autoencoders (VAEs):
- Like VAEs, DDPMs optimize a variational lower bound on the data likelihood:
  
  log p(x_0) ≥ E_q[log p_θ(x_{0:T}) - log q(x_{1:T} | x_0)]
  
- The difference lies in the Markov structure and the discretized diffusion process,
  which models transitions step-by-step instead of all at once.

#### Summary of Training and Sampling:
1. Training:
   - Sample x_0 from the dataset and t uniformly from 1,...,T
   - Generate x_t using the forward process
   - Train the model to predict the noise ε using the loss: L = ||ε - ε_θ(x_t, t)||²

2. Sampling (Generation):
   - Start with x_T ~ N(0, I)
   - Iteratively sample x_{t-1} from p_θ(x_{t-1} | x_t) until x_0 is obtained
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.unet import UNet

class DiffusionUNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256, base_filters=64, depth=5):
        super().__init__()
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # UNet with time embedding at each resolution
        self.inc = DiffusionDoubleConv(in_channels, base_filters, time_dim)
        
        # Encoder
        self.encoder = nn.ModuleList()
        in_features = base_filters
        for i in range(depth-1):
            out_features = in_features * 2
            self.encoder.append(DiffusionDown(in_features, out_features, time_dim))
            in_features = out_features
            
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(depth-1):
            self.decoder.append(DiffusionUp(in_features, in_features // 2, time_dim))
            in_features = in_features // 2
            
        # Output layer
        self.outc = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x, t, feats=None):
        # Time embedding
        t = self.time_mlp(t)
        
        # Initial convolution with time embedding
        x = self.inc(x, t)
        
        # Store skip connections
        skip_connections = [x]
        
        # Encoder path
        for i, encoder_block in enumerate(self.encoder):
            x = encoder_block(x, t)
            skip_connections.append(x)
            
            # Add features from U-Net if provided
            if feats and i < len(feats):
                x = x + feats[i+1]  # +1 because we already added the first feature
        
        # Remove the last skip connection
        x = skip_connections.pop()
        
        # Decoder path
        for decoder_block in self.decoder:
            skip = skip_connections.pop()
            x = decoder_block(x, skip, t)
            
        # Output
        return self.outc(x)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        # Project time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_channels),
            nn.GELU()
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x, t):
        h = self.conv1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(t)
        time_emb = time_emb[..., None, None]  # Add spatial dimensions
        h = h + time_emb
        
        h = self.conv2(h)
        return h

class DiffusionDown(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DiffusionDoubleConv(in_channels, out_channels, time_dim)

    def forward(self, x, t):
        x = self.maxpool(x)
        x = self.conv(x, t)
        return x

class DiffusionUp(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = DiffusionDoubleConv(in_channels, out_channels, time_dim)

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        # Adjust padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, t)

class GaussianDiffusion:
    """
    Gaussian Diffusion Process Implementation
    
    Key Equations:
    1. Forward Process: q(x_t | x_{t-1}) = N(x_t; √(α_t) x_{t-1}, β_t I)
    2. Direct Sampling: x_t = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε where ε ~ N(0, I)
    3. Training Loss: L = ||ε - ε_θ(x_t, t)||²
    """
    def __init__(self, timesteps=1000, beta_schedule='linear', beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        
        # Define beta schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == 'cosine':
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0, 0.999)
        
        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas                                  # α_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)        # ᾱ_t = ∏_{s=1}^t α_s
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)           # √(ᾱ_t)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)  # √(1 - ᾱ_t)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: Sample from q(x_t | x_0)
        
        This implements the direct sampling formula:
        x_t = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε where ε ~ N(0, I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, denoise_model, x_0, t, conditioning=None, noise=None):
        """
        Training loss calculation
        
        Implements the simplified loss: L = ||ε - ε_θ(x_t, t)||²
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Get noisy sample at timestep t
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise using model
        if conditioning is not None:
            model_input = torch.cat([x_t, conditioning], dim=1)
        else:
            model_input = x_t
            
        predicted_noise = denoise_model(model_input, t)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss


# Helper functions
def extract(a, t, shape):
    """
    Extract coefficients from a at indices t, reshape to shape
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(t.device)