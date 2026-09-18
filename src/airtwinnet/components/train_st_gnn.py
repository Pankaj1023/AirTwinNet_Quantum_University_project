import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.airtwinnet.components.st_gnn import STGNN


DATA_PATH = "data/processed/air_quality_temporal.csv"
MODEL_PATH = "artifacts/st_gnn_model.pt"


FEATURES = [
    "temperature",
    "humidity",
    "no2",
    "co",
    "o3",
    "hour",
    "day_of_week",
    "month"
]

TARGETS = [
    "pm2_5_future",
    "pm10_future"
]


SEQUENCE_LENGTH = 12
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001


def create_sequences(data):

    X = []
    y = []

    values = data[FEATURES].values
    targets = data[TARGETS].values

    for i in range(SEQUENCE_LENGTH, len(data)):

        X.append(
            values[i - SEQUENCE_LENGTH:i]
        )

        y.append(
            targets[i]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


def main():

    print("=" * 60)
    print("AirTwinNet ST-GNN Training")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    data = data.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    print("\nDataset shape:", data.shape)

    # Create sequences separately for each city
    X_all = []
    y_all = []

    for city, city_data in data.groupby("city"):

        city_data = city_data.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        if len(city_data) <= SEQUENCE_LENGTH:
            continue

        X_city, y_city = create_sequences(
            city_data
        )

        X_all.append(X_city)
        y_all.append(y_city)

        print(
            f"{city}: "
            f"{len(X_city)} sequences"
        )

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)

    print("\nSequence shape:", X.shape)
    print("Target shape:", y.shape)

    # Chronological 80/20 split
    split_index = int(len(X) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)

    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = STGNN(
        input_features=len(FEATURES),
        hidden_features=32,
        output_features=2
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\nModel:")
    print(model)

    print("\nTraining started...")

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0.0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()

            predictions = model(
                batch_X
            )

            loss = criterion(
                predictions,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() *
                len(batch_X)
            )

        epoch_loss = (
            total_loss /
            len(train_dataset)
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"- Loss: {epoch_loss:.6f}"
        )

    # Evaluation
    model.eval()

    with torch.no_grad():

        predictions = model(
            X_test
        )

        test_loss = criterion(
            predictions,
            y_test
        ).item()

    print("\nTraining completed.")

    print(
        f"Test MSE: {test_loss:.6f}"
    )

    # Save model
    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "features":
                FEATURES,

            "targets":
                TARGETS,

            "sequence_length":
                SEQUENCE_LENGTH,

            "hidden_features":
                32
        },
        MODEL_PATH
    )

    print(
        f"\nST-GNN model saved to: "
        f"{MODEL_PATH}"
    )

    print("\nST-GNN training pipeline completed.")


if __name__ == "__main__":
    main()