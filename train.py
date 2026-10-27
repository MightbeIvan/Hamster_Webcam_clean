import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load dataset
data = pd.read_csv("dataset.csv", header=None)

# Split features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Neural network (5 gestures)
model = nn.Sequential(
    nn.Linear(X.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 5)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(200):

    optimizer.zero_grad()

    outputs = model(X)
    loss = loss_fn(outputs, y)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print("Epoch:", epoch, "Loss:", loss.item())

# Save trained model
torch.save(model.state_dict(), "model.pth")

print("Model trained and saved!")