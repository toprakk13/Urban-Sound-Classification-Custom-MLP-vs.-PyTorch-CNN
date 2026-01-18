import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class UrbanSoundDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X) # (N, 1, 128, 128)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNN(nn.Module):
    def __init__(self, num_conv_layers, num_fc_layers, num_classes=10, input_shape=(1, 128, 128)):
        # input_shape: (Channels, Height, Width). Mel Spec:  (1, 128, 128), MFCC : (1, 40, 128)
        super(CNN, self).__init__()
        
        self.features = nn.Sequential()
        
        # Dynamic Conv Layers
        in_channels = input_shape[0]
        for i in range(num_conv_layers):
            out_channels = 16 * (2**i)
            self.features.add_module(f"conv{i}", nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.features.add_module(f"relu{i}", nn.ReLU())
            self.features.add_module(f"pool{i}", nn.MaxPool2d(2, 2))
            in_channels = out_channels
            
        self.flatten = nn.Flatten()
        
        # Dynamic Fully Connected Dimensions (Automatic)
        # Measuring output dimensions by giving a fake data to the model.
        # So we don't need to do mathematical calculations.
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.features(dummy_input)
            fc_input_dim = dummy_output.view(1, -1).size(1)
        
        self.classifier = nn.Sequential()
        for i in range(num_fc_layers - 1):
            fc_out = 128
            self.classifier.add_module(f"fc{i}", nn.Linear(fc_input_dim, fc_out))
            self.classifier.add_module(f"fc_relu{i}", nn.ReLU())
            fc_input_dim = fc_out
            
        # output layer
        self.classifier.add_module("output", nn.Linear(fc_input_dim, num_classes))
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def train_cnn(model, train_loader, test_loader, epochs, lr, decay_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
    loss_history = []
    acc_history = []
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
        
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
            
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
                
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
                
            running_loss += loss.item()
                
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
            
            # Test Accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
        acc = correct / total
        acc_history.append(acc)
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Test Acc: {acc:.4f}")
                
    return model, loss_history, acc_history