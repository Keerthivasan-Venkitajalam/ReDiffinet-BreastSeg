import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from config.config import Config
from models.unet import UNet
from models.rediffinet import ReDiffiNet
from data.dataset import get_dataloader
from utils.metrics import calculate_metrics
from utils.visualization import visualize_results
import numpy as np
from datetime import datetime
from tqdm import tqdm

def train_baseline_unet(config):
    """Train baseline U-Net model"""
    device = torch.device(config.device)
    
    # Create model
    model = UNet(config.unet_in_channels, config.num_classes).to(device)
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Create dataloaders
    train_loader = get_dataloader(config, mode='train')
    val_loader = get_dataloader(config, mode='val')
    
    # Training loop
    best_dice = 0.0
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            # One-hot encode masks
            masks_onehot = F.one_hot(masks, num_classes=config.num_classes).permute(0, 3, 1, 2).float()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = F.binary_cross_entropy_with_logits(outputs, masks_onehot)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        metrics_list = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                
                # One-hot encode masks
                masks_onehot = F.one_hot(masks, num_classes=config.num_classes).permute(0, 3, 1, 2).float()
                
                # Forward pass
                outputs = model(images)
                
                # Calculate metrics
                metrics = calculate_metrics(outputs, masks_onehot, config.metrics)
                metrics_list.append(metrics)
        
        # Average metrics across batches
        avg_metrics = {}
        for key in metrics_list[0]:
            if key == "avg":
                avg_metrics[key] = {}
                for metric in metrics_list[0][key]:
                    avg_metrics[key][metric] = np.mean([m[key][metric] for m in metrics_list])
            else:
                avg_metrics[key] = {}
                for metric in metrics_list[0][key]:
                    avg_metrics[key][metric] = np.mean([m[key][metric] for m in metrics_list])
        
        # Print metrics
        print(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {train_loss:.4f}")
        print(f"Validation Metrics:")
        print(f"  Avg Dice: {avg_metrics['avg']['dice']:.4f}")
        print(f"  Avg HD95: {avg_metrics['avg']['hausdorff95']:.4f}")
        
        # Save best model
        if avg_metrics['avg']['dice'] > best_dice:
            best_dice = avg_metrics['avg']['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': avg_metrics,
            }, os.path.join(os.path.dirname(__file__), 'weights', 'unet_best.pth'))
    
    return model

def train_rediffinet(config, baseline_model=None):
    """Train Re-DiffiNet model"""
    device = torch.device(config.device)
    
    # Create model
    rediffinet = ReDiffiNet(
        in_channels=config.unet_in_channels, 
        out_channels=config.num_classes,
        unet_base_filters=config.unet_base_filters,
        unet_depth=config.unet_depth,
        diffusion_steps=config.diffusion_steps,
        beta_schedule=config.beta_schedule,
        beta_start=config.beta_start,
        beta_end=config.beta_end
    ).to(device)
    
    # Load baseline model weights if provided
    if baseline_model is not None:
        rediffinet.unet.load_state_dict(baseline_model.state_dict())
    else:
        # Load from checkpoint
        if os.path.exists(os.path.join(os.path.dirname(__file__), 'weights', 'unet_best.pth')):
            checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 'weights', 'unet_best.pth'))
            rediffinet.unet.load_state_dict(checkpoint['model_state_dict'])
    
    # Freeze U-Net parameters
    for param in rediffinet.unet.parameters():
        param.requires_grad = False
    
    # Create optimizer - only for diffusion model
    optimizer = AdamW(rediffinet.diffusion_unet.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Create dataloaders
    train_loader = get_dataloader(config, mode='train')
    val_loader = get_dataloader(config, mode='val')
    
    # Training loop
    best_dice = 0.0
    for epoch in range(config.num_epochs):
        # Training
        rediffinet.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            # One-hot encode masks
            masks_onehot = F.one_hot(masks, num_classes=config.num_classes).permute(0, 3, 1, 2).float()
            
            # Get U-Net prediction
            with torch.no_grad():
                unet_pred = rediffinet.unet(images)
            
            # Calculate discrepancy loss
            loss = rediffinet.calculate_discrepancy_loss(images, unet_pred, masks_onehot)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        rediffinet.eval()
        metrics_list = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                
                # One-hot encode masks
                masks_onehot = F.one_hot(masks, num_classes=config.num_classes).permute(0, 3, 1, 2).float()
                
                # Forward pass - U-Net prediction
                unet_pred = rediffinet.unet(images)
                
                # Sample discrepancy and refine prediction
                refined_pred = rediffinet._refine_predictions(images, torch.sigmoid(unet_pred))
                
                # Calculate metrics
                unet_metrics = calculate_metrics(unet_pred, masks_onehot, config.metrics)
                refined_metrics = calculate_metrics(refined_pred, masks_onehot, config.metrics)
                
                metrics_list.append({
                    'unet': unet_metrics,
                    'refined': refined_metrics
                })
                
                # Visualize a sample
                if epoch % 5 == 0 and len(metrics_list) == 1:
                    os.makedirs(os.path.join(os.path.dirname(__file__), 'visualizations'), exist_ok=True)
                    
                    # Calculate discrepancy
                    discrepancy = rediffinet.calculate_discrepancy(unet_pred, masks_onehot)
                    
                    visualize_results(
                        images[0], 
                        masks_onehot[0], 
                        torch.sigmoid(unet_pred[0]) > 0.5, 
                        discrepancy[0],
                        save_path=os.path.join(os.path.dirname(__file__), 'visualizations', f'epoch_{epoch+1}.png')
                    )
        
        # Average metrics across batches
        avg_unet_metrics = {}
        avg_refined_metrics = {}
        
        for key in metrics_list[0]['unet']:
            if key == "avg":
                avg_unet_metrics[key] = {}
                avg_refined_metrics[key] = {}
                
                for metric in metrics_list[0]['unet'][key]:
                    avg_unet_metrics[key][metric] = np.mean([m['unet'][key][metric] for m in metrics_list])
                    avg_refined_metrics[key][metric] = np.mean([m['refined'][key][metric] for m in metrics_list])
            else:
                avg_unet_metrics[key] = {}
                avg_refined_metrics[key] = {}
                
                for metric in metrics_list[0]['unet'][key]:
                    avg_unet_metrics[key][metric] = np.mean([m['unet'][key][metric] for m in metrics_list])
                    avg_refined_metrics[key][metric] = np.mean([m['refined'][key][metric] for m in metrics_list])
        
        # Print metrics
        print(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {train_loss:.4f}")
        print(f"U-Net Validation Metrics:")
        print(f"  Avg Dice: {avg_unet_metrics['avg']['dice']:.4f}")
        print(f"  Avg HD95: {avg_unet_metrics['avg']['hausdorff95']:.4f}")
        print(f"Re-DiffiNet Validation Metrics:")
        print(f"  Avg Dice: {avg_refined_metrics['avg']['dice']:.4f}")
        print(f"  Avg HD95: {avg_refined_metrics['avg']['hausdorff95']:.4f}")
        
        # Save best model
        if avg_refined_metrics['avg']['dice'] > best_dice:
            best_dice = avg_refined_metrics['avg']['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': rediffinet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {
                    'unet': avg_unet_metrics,
                    'refined': avg_refined_metrics
                },
            }, os.path.join(os.path.dirname(__file__), 'weights', 'rediffinet_best.pth'))
    
    return rediffinet

def main():
    config = Config()
    
    # Create weights directory
    os.makedirs(os.path.join(os.path.dirname(__file__), 'weights'), exist_ok=True)
    
    # Train baseline U-Net
    print("Training baseline U-Net...")
    baseline_model = train_baseline_unet(config)
    
    # Train Re-DiffiNet
    print("Training Re-DiffiNet...")
    rediffinet = train_rediffinet(config, baseline_model)

if __name__ == "__main__":
    main()