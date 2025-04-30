import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

def dice_score(y_pred, y_true, epsilon=1e-6):
    """
    Calculate Dice Score
    
    Args:
        y_pred: Predicted mask (B, C, H, W)
        y_true: Ground truth mask (B, C, H, W)
        epsilon: Small value to avoid division by zero
        
    Returns:
        dice: Dice score
    """
    # Flatten the predictions and targets
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    intersection = (y_pred * y_true).sum()
    dice = (2. * intersection + epsilon) / (y_pred.sum() + y_true.sum() + epsilon)
    
    return dice

def hausdorff_distance_95(y_pred, y_true, percentile=95):
    """
    Calculate 95% Hausdorff Distance between binary masks
    
    Args:
        y_pred: Predicted mask (binary)
        y_true: Ground truth mask (binary)
        percentile: Percentile to use (default: 95)
        
    Returns:
        hd95: 95% Hausdorff distance
    """
    # Convert to numpy and ensure binary
    y_pred = y_pred.cpu().numpy().astype(bool)
    y_true = y_true.cpu().numpy().astype(bool)
    
    # Get distance transforms
    if y_pred.sum() > 0 and y_true.sum() > 0:
        # Distance from prediction to target
        dt_pred = distance_transform_edt(~y_pred)
        dt_true = distance_transform_edt(~y_true)
        
        pred_to_true = dt_true[y_pred]
        true_to_pred = dt_pred[y_true]
        
        hd95 = np.percentile(np.hstack([pred_to_true, true_to_pred]), percentile)
    elif y_pred.sum() > 0 or y_true.sum() > 0:
        # One is empty, one is not - maximum distance
        hd95 = np.sqrt(np.sum(np.array(y_pred.shape) ** 2))
    else:
        # Both are empty
        hd95 = 0.0
        
    return hd95

def calculate_metrics(y_pred, y_true, metrics_list=None):
    """
    Calculate metrics for segmentation
    
    Args:
        y_pred: Predicted mask (B, C, H, W), logits or probabilities
        y_true: Ground truth mask (B, C, H, W), one-hot encoded
        metrics_list: List of metrics to calculate
        
    Returns:
        metrics_dict: Dictionary of metrics
    """
    if metrics_list is None:
        metrics_list = ["dice", "hausdorff95"]
        
    # Apply sigmoid to convert logits to probabilities if needed
    if not torch.all((y_pred >= 0) & (y_pred <= 1)):
        y_pred = torch.sigmoid(y_pred)
        
    # Convert probabilities to binary
    y_pred_binary = (y_pred > 0.5).float()
    
    metrics_dict = {}
    
    # Calculate metrics for each class
    batch_size, num_classes = y_pred.shape[0], y_pred.shape[1]
    
    for c in range(num_classes):
        class_metrics = {}
        
        for metric in metrics_list:
            if metric == "dice":
                score = dice_score(y_pred_binary[:, c], y_true[:, c])
                class_metrics[metric] = score.item()
                
            elif metric == "hausdorff95":
                # Calculate for each example in batch
                hd95_scores = []
                
                for b in range(batch_size):
                    hd95 = hausdorff_distance_95(y_pred_binary[b, c], y_true[b, c])
                    hd95_scores.append(hd95)
                
                class_metrics[metric] = np.mean(hd95_scores)
        
        metrics_dict[f"class_{c}"] = class_metrics
    
    # Calculate average across classes
    avg_metrics = {}
    for metric in metrics_list:
        avg_metrics[metric] = np.mean([metrics_dict[f"class_{c}"][metric] for c in range(num_classes)])
    
    metrics_dict["avg"] = avg_metrics
    
    return metrics_dict