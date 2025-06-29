from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

EPOCHS = ...

X = load_features_dataset()  # torch.tensor di shape (N, D)
y = load_target_dataset()    # torch.tensor di shape (N,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)

model = SimpleNN(X.shape[1])
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for _ in range(EPOCHS):
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    preds = torch.argmax(model(X_test), dim=1)
    acc = accuracy_score(y_test.numpy(), preds.numpy())
    print(f"Accuratezza sul test set: {acc:.4f}")