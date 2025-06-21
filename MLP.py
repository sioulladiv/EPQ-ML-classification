import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt

#debugging
plt.ion() 

progress =[]


class ImageDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        self.features = self.data.iloc[:, :-1].values.astype('float32')
        self.labels = self.data.iloc[:, -1].values.astype('int64')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y)

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        return self.model(x)

def plot_data(x,y):
    progress.append([x,y])
    plt.plot([i[0] for i in progress],[i[1] for i in progress])
    plt.show()

def train_model(model, dataloader, criterion, optimizer, epochs):
    losses = []
    fig, ax = plt.subplots()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        epoch_loss = total_loss / len(dataloader)
        losses.append(epoch_loss)
        
        ax.clear()
        ax.plot(range(1, len(losses) + 1), losses)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss vs. Epoch')
        plt.draw()
        plt.pause(0.001) 
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

    plt.ioff()  

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    csv_file = "pixel_data_512x384.csv"
    input_size = 36864 
    hidden_size = 64
    num_classes = 12  
    batch_size = 32
    learning_rate = 0.0005 
    epochs = 400  

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    dataset = ImageDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLPClassifier(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, dataloader, criterion, optimizer, epochs)

    evaluate_model(model, dataloader)
