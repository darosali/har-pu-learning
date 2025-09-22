import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch

DATA_DIR = "data/Epilepsy2/"

class Epilepsy2Dataset(Dataset):
    def __init__(self, split="ALL"):
        
        if split.upper() == "ALL":
            train_path = os.path.join(DATA_DIR, "Epilepsy2_TRAIN.ts")
            test_path = os.path.join(DATA_DIR, "Epilepsy2_TEST.ts")
            X_train, y_train = self._load_ts(train_path)
            X_test, y_test = self._load_ts(test_path)
            self.data = torch.cat([X_train, X_test], dim=0)
            self.labels = torch.cat([y_train, y_test], dim=0)
        else:
            path = os.path.join(DATA_DIR, f"Epilepsy2_{split}.ts")
            self.data, self.labels = self._load_ts(path)
    
    def _load_ts(self, filepath):
        data = []
        labels = []
        with open(filepath, "r") as f:
            for line in f:
                if line.startswith("@") or line.strip() == "":
                    continue
                values_str, label_str = line.strip().split(":")
                values = [float(v) for v in values_str.split(",")]
                label = int(label_str)
                label = 1 - label
                data.append(values)
                labels.append(label)
        
        X = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        
        return X, y
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.data[index].unsqueeze(0), self.labels[index]