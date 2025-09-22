import matplotlib.pyplot as plt
import numpy as np

def visualize_samples(dataset, indices=[0]):
    
    plt.figure(figsize=(10, 4))
    
    for idx in indices:
        x, y = dataset[idx]
        x = x.squeeze(0).numpy()
        label = "Siezure" if y.item() == 1 else "Non-siezure"
        plt.plot(x, label=f"Sample {idx} ({label})")
        
    plt.title(f"Epilepsy2 EEG samples")
    plt.xlabel("Time (samples at 178 Hz, ~1 sec)")
    plt.ylabel("EEG Voltage (normalized units)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def make_pu_labels(y, positive_class=1, unlabeled_fraction=0.5):
    y_pu = y.copy()
    pos_idx = np.where(y == positive_class)[0]
    n_unlabeled = int(len(pos_idx) * unlabeled_fraction)
    np.random.seed(0)
    unlabeled_idx = np.random.choice(pos_idx, n_unlabeled, replace=False)
    y_pu[unlabeled_idx] = 0
    return y_pu, unlabeled_idx