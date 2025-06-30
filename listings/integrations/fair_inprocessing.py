from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

import fairlib as fl

EPOCHS = ...
BATCH_SIZE = ...

X = load_features_dataset()  # torch.tensor (N, D)
y = load_target_dataset()    # torch.tensor (N,)

X.sensitive = 'gender'

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)


model = SimpleNN(X.shape[1])


fauci_model = fl.Fauci(
    torchModel=model,
    optimizer=optim.Adam(
        model.parameters(
            model.parameters(), 
            lr=0.001
            )
        ),
    loss=nn.CrossEntropyLoss(),
    fairness_regularization="spd", # or "di" or others supported by _torch_metrics.get
    regularization_weight=0.5, # Example weight, adjust as needed
)

fauci_model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

with torch.no_grad():
    preds = torch.argmax(fauci_model.predict(X_test), dim=1)
    acc = accuracy_score(y_test.numpy(), preds.numpy())
    print(f"Accuratezza sul test set: {acc:.4f}")