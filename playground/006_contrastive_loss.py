from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from plox import Plox

from torch.utils.data import Dataset, DataLoader


# ---------------------------
# 1. Data Generation
# ---------------------------
class ToyDataset(Dataset):
    def __init__(self, num_samples):
        np.random.seed(42)
        self.class_1 = np.random.randn(num_samples // 2, 2) + np.array([2, 2])
        self.class_2 = np.random.randn(num_samples // 2, 2) + np.array([-2, -2])

        self.pairs = []
        self.labels = []

        for _ in range(num_samples):
            (c1, c2) = np.random.choice([0, 1], 2)
            C1 = [self.class_1, self.class_2][c1]
            C2 = [self.class_1, self.class_2][c2]
            p1 = C1[np.random.choice(len(C1))]
            p2 = C2[np.random.choice(len(C2))]
            label = 1 if (c1 == c2) else 0
            self.pairs.append((p1, p2))
            self.labels.append(label)

        self.pairs = np.array(self.pairs)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (p1, p2) = self.pairs[idx]
        label = self.labels[idx]
        return (
            torch.tensor(p1, dtype=torch.float32),
            torch.tensor(p2, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )


# Load dataset
dataset = ToyDataset(num_samples=200)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# ---------------------------
# 2. Embedding Model
# ---------------------------
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 2),
            # nn.BatchNorm1d(2),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.fc(x)


# ---------------------------
# 3. Contrastive Loss
# ---------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, x1, x2, y):
        d = torch.norm(x1 - x2, dim=1) ** 2
        loss = y * d + (1 - y) * torch.clamp(self.margin - d, min=0)
        return loss.mean()


# ---------------------------
# 4. Training Loop
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmbeddingNet().to(device)
criterion = ContrastiveLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
n_epochs = 120

loss_history = []

for epoch in range(n_epochs):
    epoch_loss = 0
    for (p1, p2, y) in dataloader:
        (p1, p2, y) = (p1.to(device), p2.to(device), y.to(device))

        optimizer.zero_grad()
        loss = criterion(model(p1), model(p2), y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    loss_history.append(epoch_loss)

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

# ---------------------------
# 5. Visualization of Embeddings
# ---------------------------
with torch.no_grad():
    emb1 = model(torch.tensor(dataset.class_1, dtype=torch.float32).to(device))
    emb2 = model(torch.tensor(dataset.class_2, dtype=torch.float32).to(device))

OUT_DIR = Path(__file__).with_suffix('')
OUT_DIR.mkdir(exist_ok=True)

# Plot the original dataset
with Plox() as px:
    px.a.scatter(dataset.class_1[:, 0], dataset.class_1[:, 1], label="Class 1", alpha=0.3)
    px.a.scatter(dataset.class_2[:, 0], dataset.class_2[:, 1], label="Class 2", alpha=0.3)
    px.a.legend()
    px.a.set_title("Original Dataset")
    px.f.savefig(OUT_DIR / "original.png")

with Plox() as px:
    px.a.loglog(loss_history, label="Contrastive Loss")
    px.a.set_xlabel("Epochs")
    px.a.set_ylabel("Loss")
    px.a.legend()
    px.a.set_title("Training Loss Curve")
    px.f.savefig(OUT_DIR / "loss_curve.png")

with Plox() as px:
    px.a.scatter(emb1[:, 0].cpu(), emb1[:, 1].cpu(), label="Class 1", alpha=0.3)
    px.a.scatter(emb2[:, 0].cpu(), emb2[:, 1].cpu(), label="Class 2", alpha=0.3)
    px.a.legend()
    px.a.set_title("Learned Embeddings")
    px.f.savefig(OUT_DIR / "embeddings.png")
