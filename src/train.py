"""
Train the denoising autoencoder and save weights to checkpoints/model.pth

Usage:
    python src/train.py
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from model import DenoisingAutoencoder


def add_gaussian_noise(images, noise_factor=0.3):
    noise = torch.randn_like(images) * noise_factor
    noisy = images + noise
    return torch.clamp(noisy, 0., 1.)


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    transform = transforms.ToTensor()
    train_data = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    model = DenoisingAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    noise_factor = 0.3

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)
            noisy_images = add_gaussian_noise(images, noise_factor)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = torch.mean((outputs - images) ** 2)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_data)
        print(f"epoch {epoch+1}/{num_epochs} - loss: {epoch_loss:.6f}")

    torch.save(model.state_dict(), "checkpoints/model.pth")
    print("saved weights to checkpoints/model.pth")


if __name__ == "__main__":
    main()
