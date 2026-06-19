# Denoising Autoencoder (PyTorch)

A convolutional autoencoder built from scratch in PyTorch that learns to remove
Gaussian noise from FashionMNIST images. No high-level training wrappers —
manual forward pass, manual MSE loss, manual backward/step.

## Structure

```
notebooks/   exploratory notebook (Colab/Jupyter)
src/         model definition, training script, inference script
checkpoints/ saved model weights (model.pth)
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
cd src
python train.py
```

Saves weights to `checkpoints/model.pth`.

## Run inference on your own image

```bash
cd src
python inference.py path/to/image.png
```

Saves a side-by-side comparison to `result.png`.

## Model

- **Encoder**: Conv2d -> ReLU -> MaxPool, twice (28x28 -> 32x7x7 latent)
- **Decoder**: ConvTranspose2d -> ReLU -> ConvTranspose2d -> Sigmoid (back to 28x28)
- **Loss**: manual MSE between denoised output and the original clean image
- **Noise**: Gaussian noise added on the fly during training
