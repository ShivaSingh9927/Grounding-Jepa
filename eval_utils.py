import re
import torch
import numpy as np

def extract_bbox(text):
    """
    Extracts [x1, y1, x2, y2] from text.
    Returns None if nan or no box found.
    """
    if "nan" in text.lower():
        return None
    
    # Match pattern [num, num, num, num]
    pattern = r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]'
    match = re.search(pattern, text)
    if match:
        try:
            return [float(match.group(i)) for i in range(1, 5)]
        except ValueError:
            return None
    return None

def calculate_iou(box1, box2):
    """
    box format: [x1, y1, x2, y2]
    """
    if box1 is None or box2 is None:
        return 0.0
    
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
        
    return intersection / union

def get_clinical_metrics(predictions, ground_truths):
    """
    Calculates Sensitivity, Specificity and mIOU.
    predictions: list of generated strings
    ground_truths: list of GT strings
    """
    tp, tn, fp, fn = 0, 0, 0, 0
    ious = []
    
    for pred_text, gt_text in zip(predictions, ground_truths):
        pred_box = extract_bbox(pred_text)
        gt_box = extract_bbox(gt_text)
        
        has_pred = pred_box is not None
        has_gt = gt_box is not None
        
        if has_gt:
            if has_pred:
                tp += 1
                ious.append(calculate_iou(gt_box, pred_box))
            else:
                fn += 1
                ious.append(0.0)
        else:
            if has_pred:
                fp += 1
            else:
                tn += 1
                
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mean_iou = np.mean(ious) if ious else 0.0
    
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "mean_iou": mean_iou,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }
