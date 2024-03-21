import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLP model definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return self.layers(x)
    
    def get_64_features(self, x):
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return self.layers(x)

def plot_tsne(features, labels, title):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(features)
    plt.figure(figsize=(8,8))
    for label in range(10):
        indices = labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=str(label), alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.savefig(f"{title}.png")

def train_mlp(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    print("Training complete")

def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    return f1, cm

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

# Train MLP
mlp = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)
train_mlp(mlp, train_loader, criterion, optimizer)
f1_mlp, cm_mlp = evaluate_model(mlp, test_loader)

# Prepare data for Scikit-learn models
X_train, X_test, y_train, y_test = train_test_split(mnist_train.data.numpy().reshape(-1, 28*28), mnist_train.targets.numpy(), test_size=0.2, random_state=42)
untrained_model = MLP().to(device)

images, labels = next(iter(test_loader))
images, labels = images.view(images.size(0), -1), labels.numpy()

with torch.no_grad():
    trained_features = mlp.get_64_features(images.to(device)).cpu().numpy()
    untrained_features = untrained_model.get_64_features(images.to(device)).cpu().numpy()

plot_tsne(trained_features, labels, "t-SNE of the 64-neuron layer output (Trained Model)")
plot_tsne(untrained_features, labels, "t-SNE of the 64-neuron layer output (Untrained Model)")

# Train Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
cm_rf = confusion_matrix(y_test, y_pred_rf)

from sklearn.preprocessing import StandardScaler

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=500, solver='lbfgs', multi_class='auto')
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
f1_log_reg = f1_score(y_test, y_pred_log_reg, average='weighted')
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)

# Print F1-scores and Confusion Matrices
print("MLP F1-score:", f1_mlp)
print("MLP Confusion Matrix:\n", cm_mlp)
print("Random Forest F1-score:", f1_rf)
print("Random Forest Confusion Matrix:\n", cm_rf)
print("Logistic Regression F1-score:", f1_log_reg)
print("Logistic Regression Confusion Matrix:\n", cm_log_reg)
def print_layer_neurons(model):
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            print(f"Layer: {layer}, Neurons in: {layer.in_features}, Neurons out: {layer.out_features}")

print_layer_neurons(mlp)


fashion_mnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
fashion_mnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
fashion_train_loader = DataLoader(fashion_mnist_train, batch_size=64, shuffle=True)
fashion_test_loader = DataLoader(fashion_mnist_test, batch_size=64, shuffle=False)

f1_fashion_mlp, cm_fashion_mlp = evaluate_model(mlp, fashion_test_loader)

# Step 4: Print the model's performance on the Fashion-MNIST dataset
print("Fashion-MNIST MLP F1-score:", f1_fashion_mlp)
print("Fashion-MNIST MLP Confusion Matrix:\n", cm_fashion_mlp)


images, labels = next(iter(fashion_test_loader))
images, labels = images.view(images.size(0), -1), labels.numpy()

with torch.no_grad():
    fashion_trained_features = mlp.get_64_features(images.to(device)).cpu().numpy()
    
plot_tsne(fashion_trained_features, labels, "t-SNE of the 64-neuron layer output (Fashion Trained Model)")

