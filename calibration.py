import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils import compute_class_weight

# https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits, _ = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature
        
    def set_temperature(self, logits, labels):
        self.eval()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        device = next(self.parameters()).device
        logits = logits.to(device)
        labels = labels.to(device)
        
        nll_criterion = nn.CrossEntropyLoss()
        #optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=500)
        optimizer = torch.optim.AdamW([self.temperature], lr=0.01)

        for _ in range(1500):
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            optimizer.step()

        # def eval():
        #     optimizer.zero_grad()
        #     loss = nll_criterion(self.temperature_scale(logits), labels)
        #     loss.backward()
        #     return loss

        # optimizer.step(eval)
        print(f"Optimal temperature: {self.temperature.item():.3f}")
        return self

def find_temperature_grid(logits, y_true, n_temps=200):
    best_T, best_ece = None, float('inf')
    for T_candidate in np.linspace(0.0, 3.0, num=n_temps):
        scaled_logits = logits / T_candidate
        probs = torch.softmax(torch.tensor(scaled_logits), dim=1).numpy()
        ece = expected_calibration_error(y_true, probs[:, 1])  # For binary case
        if ece < best_ece:
            best_ece = ece
            best_T = T_candidate
    print(f"Best T: {best_T}, Best ECE: {best_ece}")
    return best_T

def expected_calibration_error(probs, labels, n_bins=10):
    """
    Compute Expected Calibration Error (ECE)
    
    Args:
        probs (np.array): predicted probabilities (shape: [n_samples, n_classes])
        labels (np.array): true labels (shape: [n_samples])
        n_bins (int): number of bins
    
    Returns:
        float: ECE score
    """
    # Get predicted class and confidence
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    # Bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Find indices of samples in the current bin
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i+1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if np.any(in_bin):
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * np.sum(in_bin) / len(labels)
    
    return ece


def adaptive_calibration_error(probs, labels, n_bins=10):
    """
    Computes Adaptive Calibration Error (ACE) as defined in the original paper.

    Args:
        probs (np.ndarray): Predicted probabilities, shape (n_samples, n_classes)
        labels (np.ndarray): True labels, shape (n_samples,)
        n_bins (int): Number of bins for probability ranges (ranges R)

    Returns:
        ace (float): Adaptive Calibration Error
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    n_samples, n_classes = probs.shape
    
    # One-hot encode labels
    one_hot_labels = np.eye(n_classes)[labels]
    
    ace = 0.0
    
    # For each class
    for k in range(n_classes):
        class_probs = probs[:, k]
        class_labels = one_hot_labels[:, k]
        
        # Compute bin edges (adaptive bins)
        sorted_indices = np.argsort(class_probs)
        sorted_probs = class_probs[sorted_indices]
        sorted_labels = class_labels[sorted_indices]
        
        bin_size = int(np.ceil(n_samples / n_bins))
        
        # For each bin
        for r in range(n_bins):
            start = r * bin_size
            end = min((r + 1) * bin_size, n_samples)
            if start >= end:
                continue
            bin_probs = sorted_probs[start:end]
            bin_labels = sorted_labels[start:end]
            
            avg_conf = np.mean(bin_probs)
            avg_acc = np.mean(bin_labels)
            
            ace += np.abs(avg_conf - avg_acc)
    
    # Normalize by K * R
    ace /= (n_classes * n_bins)
    
    return ace

def ece_breakdown(y_true, probs, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) and return per-bin breakdown rows.

    Args:
        y_true (np.ndarray): True labels, shape (n_samples,)
        probs (np.ndarray): Predicted probabilities, shape (n_samples, n_classes)
        n_bins (int): Number of bins

    Returns:
        ece (float): Expected Calibration Error
        rows (list of tuples): Each row as (bin_idx, n, conf_mean, acc_mean, abs_diff, weighted)
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    total_samples = len(y_true)
    ece = 0.0
    rows = []

    for i in range(n_bins):
        bin_lower, bin_upper = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        n = np.sum(in_bin)
        if n == 0:
            continue

        acc_mean = np.mean(predictions[in_bin] == y_true[in_bin])
        conf_mean = np.mean(confidences[in_bin])
        abs_diff = abs(acc_mean - conf_mean)
        weighted = abs_diff * (n / total_samples)
        ece += weighted

        rows.append((i, n, bin_lower, bin_upper, conf_mean, acc_mean, abs_diff, weighted))

    return ece, rows


def ace_breakdown(y_true, probs, n_bins=10):
    """
    Compute Adaptive Calibration Error (ACE) as defined in the paper and
    return per-bin breakdown rows.

    Args:
        y_true (np.ndarray): True labels, shape (n_samples,)
        probs (np.ndarray): Predicted probabilities, shape (n_samples, n_classes)
        n_bins (int): Number of bins

    Returns:
        ace (float): Adaptive Calibration Error
        rows (list of tuples): Each row as (bin_idx, n, lo, hi, conf_mean, acc_mean, abs_diff)
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    n_samples, n_classes = probs.shape
    one_hot_labels = np.eye(n_classes)[y_true]
    
    ace = 0.0
    rows = []

    for k in range(n_classes):
        class_probs = probs[:, k]
        class_labels = one_hot_labels[:, k]

        sorted_idx = np.argsort(class_probs)
        sorted_probs = class_probs[sorted_idx]
        sorted_labels = class_labels[sorted_idx]
        bin_size = int(np.ceil(n_samples / n_bins))

        for r in range(n_bins):
            start = r * bin_size
            end = min((r + 1) * bin_size, n_samples)
            if start >= end:
                continue

            bin_probs = sorted_probs[start:end]
            bin_labels = sorted_labels[start:end]

            lo, hi = bin_probs[0], bin_probs[-1]
            conf_mean = np.mean(bin_probs)
            acc_mean = np.mean(bin_labels)
            abs_diff = abs(conf_mean - acc_mean)
            ace += abs_diff

            rows.append((k, r, len(bin_probs), lo, hi, conf_mean, acc_mean, abs_diff))

    ace /= (n_classes * n_bins)
    return ace, rows

