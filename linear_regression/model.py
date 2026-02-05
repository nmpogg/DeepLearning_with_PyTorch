import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from matplotlib import pyplot as plt

torch.manual_seed(42)

X = torch.randn(500, 5)      # (500 samples, 5 features)
true_weights = torch.tensor([2.123, 3.123, 4.123, 5.123, 6.123]).reshape(5, 1)  # (5 features, 1 output)

y = X @ true_weights + torch.tensor([3.324]) + 0.1 * torch.randn(500, 1)

dataset = TensorDataset(X, y)

train_dataset, val_dataset = random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class LinearRegression(nn.Module):
    def __init__(self, features, output):
        super().__init__()
        self.linear = nn.Linear(features, output) # features input features, output output features

    def forward(self, data):
        return self.linear(data)
    

model = LinearRegression(5, 1)
for name, param in model.named_parameters():
    print(name, param.data)

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.03, weight_decay=0.01)

num_epochs = 100

loss_history = {'train': [], 'val': []}
for epoch in range(num_epochs):
    train_loss_avg = 0.0
    for X_batch, y_batch in train_loader:
        y_hat = model(X_batch)
        loss = criterion(y_hat, y_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss_avg += loss.item()
    train_loss_avg /= len(train_loader)
    loss_history['train'].append(train_loss_avg)

    model.eval()
    val_loss_avg = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            y_val_hat = model(X_val)
            val_loss = criterion(y_val_hat, y_val)
            val_loss_avg += val_loss.item()
        val_loss_avg /= len(val_loader)
        loss_history['val'].append(val_loss_avg)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], train_loss: {train_loss_avg:.4f}, val_loss: {val_loss_avg:.4f}")

w = model.linear.weight.data.numpy()
b = model.linear.bias.data.numpy()

print(f"Learned weight:  {w}")
print(f"Learned bias: {b}")

predicted = model(X).detach().numpy()
plt.figure(figsize=(10, 5))

plt.plot(loss_history['train'], label='Train Loss')
plt.plot(loss_history['val'], label='Validation Loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()

