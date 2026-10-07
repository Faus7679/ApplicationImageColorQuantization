"""Color quantization of an image with k-means clustering (pure NumPy + Pillow)."""
import argparse
import os

import numpy as np
from PIL import Image


def kmeans(pixels, k, max_iter=50, tol=1e-4, seed=0):
    """Cluster an (N, 3) float array into k clusters. Returns (centers, labels)."""
    rng = np.random.default_rng(seed)
    k = min(k, len(pixels))
    centers = pixels[rng.choice(len(pixels), k, replace=False)].copy()
    labels = np.zeros(len(pixels), dtype=np.int64)
    for _ in range(max_iter):
        # assignment step: nearest center (squared Euclidean distance)
        dists = ((pixels ** 2).sum(1)[:, None] - 2 * pixels @ centers.T
                 + (centers ** 2).sum(1)[None, :])
        labels = dists.argmin(axis=1)
        # update step: mean of each cluster
        new_centers = centers.copy()
        for j in range(k):
            members = pixels[labels == j]
            if len(members):
                new_centers[j] = members.mean(axis=0)
        shift = np.abs(new_centers - centers).max()
        centers = new_centers
        if shift < tol:
            break
    return centers, labels


def quantize(image, k=16, **kw):
    """Return (quantized PIL image, palette, labels) for an RGB PIL image."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    h, w, _ = arr.shape
    centers, labels = kmeans(arr.reshape(-1, 3), k, **kw)
    palette = np.clip(np.rint(centers), 0, 255).astype(np.uint8)
    out = palette[labels].reshape(h, w, 3)
    return Image.fromarray(out, "RGB"), palette, labels.reshape(h, w)


def save_compressed(labels, palette, path):
    """Save as a palette (indexed) PNG: k colors + one index per pixel."""
    img = Image.fromarray(labels.astype(np.uint8), "P")
    img.putpalette(palette.flatten().tolist())
    img.save(path, optimize=True)


def make_demo_image(size=256):
    """Synthetic gradient image with shapes, used when no input is given."""
    y, x = np.mgrid[0:size, 0:size]
    img = np.stack([x, y, 255 - (x + y) // 2], axis=-1).astype(np.uint8)
    r = (x - size // 2) ** 2 + (y - size // 2) ** 2 < (size // 4) ** 2
    img[r] = (230, 200, 30)
    return Image.fromarray(img, "RGB")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", nargs="?", help="input image (demo image if omitted)")
    p.add_argument("-k", type=int, default=16, help="number of colors")
    p.add_argument("-o", "--output", default="quantized.png")
    a = p.parse_args()
    if a.k < 1 or a.k > 256:
        p.error("k must be between 1 and 256")

    img = Image.open(a.input) if a.input else make_demo_image()
    img = img.convert("RGB")
    orig_path = "original.png"
    if not a.input:
        img.save(orig_path)
    q, palette, labels = quantize(img, a.k)
    save_compressed(labels, palette, a.output)

    a_ = np.asarray(img, dtype=np.float64)
    b_ = np.asarray(q, dtype=np.float64)
    mse = ((a_ - b_) ** 2).mean()
    psnr = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    print(f"colors in original : {len(np.unique(a_.reshape(-1, 3), axis=0))}")
    print(f"colors (k)         : {len(palette)}")
    print(f"MSE / PSNR         : {mse:.2f} / {psnr:.2f} dB")
    print(f"raw size           : {a_.size} bytes (24 bpp)")
    print(f"quantized file     : {os.path.getsize(a.output)} bytes -> {a.output}")


if __name__ == "__main__":
    main()
