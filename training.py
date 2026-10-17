import torch
import os
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from dataset import BattleShipDataset, write_training_log
from model import BattleShipModel
from sklearn.model_selection import train_test_split


if "__main__" == __name__:

    device = "cuda"
    data_dir = os.path.abspath("data")

    ds = BattleShipDataset(data_dir=data_dir)

    ds_size = len(ds)

    indices = list(range(ds_size))
    batch_size = 64
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=12)

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    model = BattleShipModel().to(device=device)
    n_epochs = 15
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters())

    for t in range(n_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            write_training_log(f"Epoch {t+1}\n-------------------------------")
            model.train()
            size = len(train_ds)
            for batch, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                y_pred = model(inputs)
                illegal = inputs.flatten(start_dim=1) >= 7
                y_pred = y_pred.masked_fill(illegal, float("-inf"))
                loss = criterion(y_pred, labels)
                # print(y_pred.shape, labels.shape, labels.dtype)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * 32 + len(inputs)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    write_training_log(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            model.eval()
            size = len(val_loader.dataset)
            num_batches = len(val_loader)
            test_loss = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    pred = model(inputs)
                    illegal = inputs.flatten(start_dim=1) >= 7
                    pred = pred.masked_fill(illegal, float("-inf"))
                    test_loss += criterion(pred, labels).item()
                test_loss /= num_batches
                print(f"Validation loss: {test_loss:>8f}\n")
                write_training_log(f"Validation loss: {test_loss:>8f}\n")
            checkpoint = {
                "epoch": t + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss
                    }
            torch.save(checkpoint, f"checkpoints/checkpoint_{t + 1}.pth")
    torch.save(model.state_dict(), "models/BattleShipAgentV2.pth")