import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

input_dim = X_scaled.shape[1]

inputs = Input(shape=(input_dim,))
encoded = Dense(16, activation="relu")(inputs)
encoded = Dense(8, activation="relu")(encoded)

decoded = Dense(16, activation="relu")(encoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inputs, decoded)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.fit(
    X_scaled,
    X_scaled,
    epochs=5,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

autoencoder.save("models/autoencoder.keras")

print("Autoencoder saved successfully!")