class Config:
    # Data configuration
    data_path = "path/to/data"
    batch_size = 4
    num_workers = 4
    image_size = (256, 256)  # (H, W) for 2D breast images
    
    # Breast tumor segmentation specific
    # Different from BraTS which has 3 tumor regions
    # For breast we'll use: 1) Mass, 2) Calcification, 3) Architectural Distortion
    num_classes = 3
    
    # U-Net configuration
    unet_in_channels = 1  # For mammogram, use 1; for MRI, could be multiple
    unet_base_filters = 64
    unet_depth = 5
    
    # Diffusion model configuration
    diffusion_steps = 1000
    beta_schedule = "linear"
    beta_start = 0.0001
    beta_end = 0.02
    
    # Training configuration
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-4
    
    # Metrics
    metrics = ["dice", "hausdorff95"]
    
    # Device
    device = "cuda"
    
    # Cross-validation
    num_folds = 5