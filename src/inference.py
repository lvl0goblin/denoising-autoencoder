"""
Load trained weights and denoise an image.

Usage:
    python src/inference.py path/to/image.png
"""
import sys
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import DenoisingAutoencoder


def load_model(weights_path="checkpoints/model.pth", device="cpu"):
    model = DenoisingAutoencoder()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model.to(device)


def denoise_image(model, image_path, device="cpu"):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path)
    tensor = transform(img).unsqueeze(0).to(device)  # (1, 1, 28, 28)

    with torch.no_grad():
        output = model(tensor)

    return tensor.cpu().squeeze().numpy(), output.cpu().squeeze().numpy()


def main():
    if len(sys.argv) < 2:
        print("usage: python src/inference.py path/to/image.png")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device=device)
    original, denoised = denoise_image(model, sys.argv[1], device=device)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("input")
    axes[0].axis("off")

    axes[1].imshow(denoised, cmap="gray")
    axes[1].set_title("denoised")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("result.png")
    print("saved result.png")


if __name__ == "__main__":
    main()
