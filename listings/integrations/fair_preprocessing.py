import fairlib as fl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X = load_features_dataset()
y = load_target_dataset()

X.sensitive = 'gender'  # Example sensitive attribute
EPOCHS = ...
BATCH_SIZE = ...

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

layers_shape = X_train.shape[1]

lfr = fl.LFR(
    input_dim=layers_shape,
    latent_dim=8,
    output_dim=layers_shape,
    alpha_z=1.0,
    alpha_x=1.0,
    alpha_y=1.0,
)

lfr.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

X_train_transformed = lfr.transform(X_train)
X_test_transformed = lfr.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train_transformed, y_train)

y_pred = classifier.predict(X_test_transformed)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuratezza sul test set: {accuracy:.4f}")