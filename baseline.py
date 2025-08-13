import torch
import numpy
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve, precision_score, precision_recall_curve
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pu_loss import PULoss, PURankingLoss
from mamba_ssm import Mamba

class Conv_Net(nn.Module):
    def __init__(self, in_channel, seq_length, hidden_dim, n_output):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (seq_length // 2), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_output)
        
        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.maxpool,
            self.dropout
        )
        
        self.classifier = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2
        )
        
    def forward(self, x):
        x_features = self.features(x)
        x_flattened = self.flatten(x_features)
        y_out = self.classifier(x_flattened)
        return y_out
    
class LSTMNet(nn.Module):
    def __init__(self, in_channel, hidden_dim, n_output, num_layers=1):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channel, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        #self.fc = nn.Linear(hidden_dim, n_output)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, n_output)
        )

    def forward(self, x):
        # x: (batch, seq_length, in_channel)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq_length, hidden_dim)
        # Use last timestep
        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim)
        last_output = self.bn(last_output)
        out = self.dropout(last_output)
        out = self.fc(last_output)
        return out

# class MambaClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64, num_classes=1):
#         super().__init__()
#         self.input_proj = nn.Linear(input_dim, hidden_dim)
#         self.mamba = Mamba(d_model=hidden_dim)
#         self.norm = nn.LayerNorm(hidden_dim)
#         self.pool = nn.AdaptiveAvgPool1d(1) 
#         self.fc = nn.Linear(hidden_dim, num_classes)
        
#     def forward(self, x):
#         # x: (batch, seq_len, input_dim)
#         x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
#         x = self.mamba(x)       # (batch, seq_len, hidden_dim)
#         x = self.norm(x)
#         x = x.transpose(1, 2)   # For pooling: (batch, hidden_dim, seq_len)
#         x = self.pool(x).squeeze(-1)  # (batch, hidden_dim)
#         x = self.fc(x)          # (batch, num_classes)
#         return x

class MambaBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.mamba = Mamba(d_model=hidden_dim)  
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mamba(x)
        x = self.dropout(x)
        return self.norm(x) 


class MambaClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=1, num_layers=4, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of Mamba residual blocks
        self.blocks = nn.Sequential(
            *[MambaBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )

        self.pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Input: (batch, seq_len, input_dim)
        x = self.input_proj(x)           # (batch, seq_len, hidden_dim)
        x = self.blocks(x)               # stacked Mamba layers
        x = x.transpose(1, 2)            # (batch, hidden_dim, seq_len)
        x = self.pool(x).squeeze(-1)     # (batch, hidden_dim)
        return self.fc(x)                # (batch, num_classes)
    
    
def train(model, criterion, train_loader, test_loader, lr, epochs, device):
    
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        n_train_acc, n_test_acc = 0, 0
        train_loss, test_loss = 0, 0
        n_train, n_test = 0, 0
        
        model.train()
        for inputs, labels in tqdm(train_loader):
            # print("Batch unique labels:", labels.unique())
            # assert labels.min() >= 0 and labels.max() < n_output, "Invalid label detected!"
            if isinstance (model, (LSTMNet, MambaClassifier)):
                inputs = inputs.permute(0, 2, 1)
                
            train_batch_size = len(labels) 
            n_train += train_batch_size
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            #print(outputs.shape)
            #break

            if isinstance(criterion, PULoss):
        
                loss = criterion(outputs.view(-1), labels)
                preds = torch.where(
                    outputs > 0,
                    torch.tensor(1, device=outputs.device),
                    torch.tensor(-1, device=outputs.device)
                ).view(-1) 
            else:
                loss = criterion(outputs, labels)
                preds = torch.max(outputs, 1)[1]
            #break
            loss.backward()
            optimizer.step() 
                   
            train_loss += loss.item() * train_batch_size 
            n_train_acc += (preds == labels).sum().item()
        
        model.eval()
        
        for inputs_test, labels_test in test_loader:
            if isinstance (model, (LSTMNet, MambaClassifier)):
                inputs_test = inputs_test.permute(0, 2, 1)
            test_batch_size = len(labels_test)
            n_test += test_batch_size
            
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)
            
            outputs_test = model(inputs_test)
            
            if isinstance(criterion, PULoss):
                loss_test = criterion(outputs_test.view(-1), labels_test)
                preds_test = torch.where(
                    outputs_test > 0,
                    torch.tensor(1, device=outputs_test.device),
                    torch.tensor(-1, device=outputs_test.device)
                ).view(-1)
            else:
                loss_test = criterion(outputs_test, labels_test)
                preds_test = torch.max(outputs_test, 1)[1]
            
            test_loss += loss_test.item() * test_batch_size
            n_test_acc += (preds_test == labels_test).sum().item()
        
        train_acc = n_train_acc / n_train
        test_acc = n_test_acc / n_test
        avg_train_loss = train_loss / n_train
        avg_test_loss = test_loss / n_test
        print (f'Epoch [{(epoch+1)}/{epochs}], tr_loss: {avg_train_loss:.5f} tr_acc: {train_acc:.5f} test_loss: {avg_test_loss:.5f}, test_acc: {test_acc:.5f}')
    return model