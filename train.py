# train.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from models.resnet_bilstm import HybridResNetBiLSTM


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def normalize_timeseries(windows):
    mean = windows.mean(dim=2, keepdim=True)
    std = windows.std(dim=2, keepdim=True) + 1e-6
    return (windows - mean) / std


def normalize_stats(stats):
    mean = stats.mean(dim=0, keepdim=True)
    std = stats.std(dim=0, keepdim=True) + 1e-6
    return (stats - mean) / std


def train_model():

    set_seed(42)

    print("Loading dataset...")

    windows = np.load("dataset_windows.npy")
    stats = np.load("dataset_stats.npy")
    labels = np.load("dataset_rul.npy")

    windows = torch.tensor(windows).permute(0, 2, 1).float()
    stats = torch.tensor(stats).float()
    labels = torch.tensor(labels).float().unsqueeze(1)

    windows = normalize_timeseries(windows)
    stats = normalize_stats(stats)

    dataset = TensorDataset(windows, stats, labels)

    indices = np.arange(len(dataset))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    device = torch.device("cpu")

    model = HybridResNetBiLSTM(stat_dim=56)
    model.to(device)

    criterion = nn.HuberLoss(delta=0.05)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=4
    )

    best_val_loss = float("inf")
    early_stop_patience = 12
    no_improve = 0

    for epoch in range(60):

        model.train()
        train_loss = 0.0

        for x, s, y in train_loader:
            x, s, y = x.to(device), s.to(device), y.to(device)

            optimizer.zero_grad()

            outputs = model(x, s)
            loss = criterion(outputs, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        preds = []
        labels_list = []

        with torch.no_grad():
            for x, s, y in val_loader:
                x, s, y = x.to(device), s.to(device), y.to(device)

                outputs = model(x, s)
                loss = criterion(outputs, y)
                val_loss += loss.item()

                preds.append(outputs.cpu().numpy())
                labels_list.append(y.cpu().numpy())

        val_loss /= len(val_loader)

        preds = np.vstack(preds).flatten()
        labels_list = np.vstack(labels_list).flatten()

        mae = mean_absolute_error(labels_list, preds)
        rmse = np.sqrt(np.mean((preds - labels_list) ** 2))

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1:02d} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"MAE: {mae:.4f} | "
            f"RMSE: {rmse:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_rul_model.pth")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            print("\nEarly stopping triggered.")
            break

    print("\nBest Validation Loss:", best_val_loss)


if __name__ == "__main__":
    train_model()
