import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.airtwinnet.components.st_gnn import STGNN


DATA_PATH = "data/processed/air_quality_temporal.csv"
MODEL_PATH = "artifacts/st_gnn_model_v2.pt"

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
HIDDEN_FEATURES = 32


def create_sequences(data):

    X = []
    y = []

    feature_values = data[FEATURES].values
    target_values = data[TARGETS].values

    for i in range(SEQUENCE_LENGTH, len(data)):
        X.append(
            feature_values[
                i - SEQUENCE_LENGTH:i
            ]
        )

        y.append(
            target_values[i]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


def main():

    print("=" * 60)
    print("AirTwinNet ST-GNN V2 Training")
    print("=" * 60)

    data = pd.read_csv(DATA_PATH)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    data = data.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    train_X = []
    train_y = []

    test_X = []
    test_y = []

    print("\nCity-wise chronological split")
    print("-" * 60)

    for city, city_data in data.groupby("city"):

        city_data = city_data.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        X_city, y_city = create_sequences(
            city_data
        )

        split_index = int(
            len(X_city) * 0.8
        )

        X_train_city = X_city[:split_index]
        y_train_city = y_city[:split_index]

        X_test_city = X_city[split_index:]
        y_test_city = y_city[split_index:]

        train_X.append(X_train_city)
        train_y.append(y_train_city)

        test_X.append(X_test_city)
        test_y.append(y_test_city)

        print(
            f"{city}: "
            f"Train={len(X_train_city)}, "
            f"Test={len(X_test_city)}"
        )

    train_X = np.concatenate(train_X)
    train_y = np.concatenate(train_y)

    test_X = np.concatenate(test_X)
    test_y = np.concatenate(test_y)

    print("\nFinal dataset")
    print("-" * 60)

    print("Training samples:", len(train_X))
    print("Testing samples:", len(test_X))

    print("Training shape:", train_X.shape)
    print("Testing shape:", test_X.shape)

    X_train = torch.tensor(
        train_X,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        train_y,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        test_X,
        dtype=torch.float32
    )

    y_test = torch.tensor(
        test_y,
        dtype=torch.float32
    )

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = STGNN(
        input_features=len(FEATURES),
        hidden_features=HIDDEN_FEATURES,
        output_features=2
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\nModel")
    print("-" * 60)
    print(model)

    print("\nTraining started...")

    for epoch in range(EPOCHS):

        model.train()

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
                loss.item()
                * len(batch_X)
            )

        epoch_loss = (
            total_loss
            / len(train_dataset)
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"- Loss: {epoch_loss:.6f}"
        )

    model.eval()

    with torch.no_grad():

        test_predictions = model(
            X_test
        )

        test_loss = criterion(
            test_predictions,
            y_test
        ).item()

    print("\nTraining completed.")
    print(
        f"Test MSE: {test_loss:.6f}"
    )

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_features": len(FEATURES),
        "hidden_features": HIDDEN_FEATURES,
        "output_features": 2,
        "features": FEATURES,
        "targets": TARGETS,
        "sequence_length": SEQUENCE_LENGTH,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE
    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )

    print(
        f"ST-GNN V2 model saved to: "
        f"{MODEL_PATH}"
    )

    print("\nST-GNN V2 training completed successfully.")


if __name__ == "__main__":
    main()